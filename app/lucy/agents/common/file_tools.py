import csv
import io
import mimetypes
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

import pandas as pd
import requests
from docx import Document
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from lucy.agents.clients.openai_client import get_openai_client
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps


# ------------------------------------------------------------
# Structured output schema
# ------------------------------------------------------------


class GenericFileAnalysis(BaseModel):
    """Generic analysis for any file that can be turned into text."""

    file_name: str = Field(description="Original file name")
    file_type: str = Field(
        description="High level type, for example: pdf, csv, excel, docx, text, image, audio, video"
    )
    summary: str = Field(
        description="Comprehensive summary of the file content. Should be one paragraph for smaller files, two paragraphs for larger/complex files. Focus on key insights, main content, and important findings."
    )
    key_topics: List[str] = Field(
        description="Main topics or sections covered in the file"
    )
    important_entities: List[str] = Field(
        description="Key people, companies, products, places, or concepts mentioned in the file"
    )
    potential_actions: List[str] = Field(
        description="Recommended next steps or actions a human might take based on this file"
    )


# No default model — callers pass model per-request via Agent.run(model=...)
_file_analysis_agent = Agent(
    system_prompt="You are a file analysis assistant. Return structured output only.",
    output_type=GenericFileAnalysis,
)


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _render_csv_for_llm(file_bytes: bytes, max_rows: int = 50) -> str:
    """Render CSV content into a compact textual form for the model."""
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        return "CSV file appears to be empty."

    header = rows[0]
    sample = rows[1 : max_rows + 1]

    lines: List[str] = []
    lines.append("CSV structure")
    lines.append("Columns: " + ", ".join(header))
    lines.append("")
    lines.append(f"Sample rows (up to {max_rows} rows):")

    for r in sample:
        row_str = " | ".join(r)
        if len(row_str) > 500:
            row_str = row_str[:500] + "..."
        lines.append(row_str)

    return "\n".join(lines)


def _render_excel_for_llm(file_bytes: bytes, max_rows: int = 30) -> str:
    """Render Excel workbook into text: sheet names plus sample rows."""
    buf = io.BytesIO(file_bytes)
    try:
        # sheet_name=None returns dict sheet_name -> DataFrame
        xls = pd.read_excel(buf, sheet_name=None)
    except Exception as e:
        return f"Failed to parse Excel file: {e}"

    if not xls:
        return "Excel workbook appears to be empty."

    parts: List[str] = []
    for sheet_name, df in xls.items():
        parts.append(f"Sheet: {sheet_name}")
        if df.empty:
            parts.append("(empty sheet)")
            parts.append("")
            continue

        df_sample = df.head(max_rows)
        csv_str = df_sample.to_csv(index=False)
        parts.append(csv_str)
        parts.append("")

    text = "\n".join(parts)
    max_chars = 200_000
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _render_docx_for_llm(file_bytes: bytes, max_chars: int = 200_000) -> str:
    """Extract readable text from a DOCX file."""
    buf = io.BytesIO(file_bytes)
    try:
        doc = Document(buf)
    except Exception as e:
        return f"Failed to parse DOCX file: {e}"

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        return "DOCX document appears to be empty."

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _render_generic_text_for_llm(file_bytes: bytes, max_chars: int = 200_000) -> str:
    """Fallback renderer for unknown text like formats."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return "File is not readable as UTF-8 text."

    if not text.strip():
        return "File appears to be empty."

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


# ------------------------------------------------------------
# Public registration function
# ------------------------------------------------------------


def register_file_analysis_tool(agent: Agent) -> None:
    """
    Register file_analysis_tool on the given Agent.

    Assumes the agent uses deps_type=ChatDeps so ctx: RunContext[ChatDeps] is valid.
    The tool:
      - takes the first attachment from ctx.deps.attachments
      - derives file name and extension from attachment metadata and URL
      - for PDF uses OpenAI file context stuffing via input_file
      - for CSV, Excel, DOCX and other text formats renders content to text and sends as input_text
      - returns a dict with success flag, model_used, file metadata and GenericFileAnalysis payload
    """

    @agent.tool
    async def file_analysis_tool(
        ctx: RunContext[ChatDeps],
        user_question: Optional[str] = Field(
            default=None,
            description="Optional specific question the user wants answered based on the file content. If provided, focus on answering this question. If not provided, provide generic marketing/advertising analysis and suggestions.",
        ),
    ) -> Dict[str, Any]:
        """
        Analyze a single file from the current request attachments and provide a comprehensive summary.

        Primary focus: Analyze the file content and provide a detailed summary. If the user asks a specific question,
        answer it based on the file content. If no question is asked, provide generic marketing/advertising analysis
        and actionable suggestions.

        Model selection:
          - PDF files: Uses Models.FILE_ANALYSIS_PDF (for PDF context stuffing via input_file)
          - All other files: Uses Models.FILE_ANALYSIS_DEFAULT (CSV, Excel, DOCX, text files, etc.)

        Behavior:
          - If the file is a PDF, upload and pass as input_file (PDF context stuffing).
          - For CSV, parse columns and sample rows to text.
          - For Excel, read sheets and sample rows to text.
          - For DOC or DOCX, extract paragraphs to text.
          - For anything else, decode to UTF-8 text as best effort.
        In all non PDF cases the content is sent as input_text only.
        Returns a structured GenericFileAnalysis payload with file summary and analysis.
        """
        ctx.deps.status_queue.put_nowait("Analyzing your file")
        try:
            attachments = ctx.deps.attachments or []
            if not attachments:
                raise ValueError("No attachments found on this request to analyze.")

            attachment = attachments[0]
            if not isinstance(attachment, dict):
                raise ValueError(
                    "Attachment format is not supported. Expected a dict payload."
                )

            url = attachment.get("url")
            if not url:
                raise ValueError("Attachment is missing url field.")

            # Prefer explicit name first
            file_name = attachment.get("file_name") or attachment.get("name") or ""

            # Derive name from URL path
            parsed = urlparse(url)
            path_part = unquote(parsed.path or "")
            url_name = os.path.basename(path_part) or "uploaded_file"

            # If explicit name has no extension but URL has one, prefer URL
            file_name_ext = os.path.splitext(file_name)[1]
            url_name_ext = os.path.splitext(url_name)[1]
            if not file_name or (not file_name_ext and url_name_ext):
                file_name = url_name

            if not file_name:
                file_name = "uploaded_file"

            ext = os.path.splitext(file_name)[1].lower()

            # Derive MIME from extension if needed
            file_type = (attachment.get("file_type") or "").lower()
            mime_from_ext, _ = mimetypes.guess_type(file_name)
            if not file_type:
                file_type = mime_from_ext or "application/octet-stream"

            # Select model based on file type: PDF vs everything else
            is_pdf = ext == ".pdf"
            model = Models.FILE_ANALYSIS_PDF if is_pdf else Models.FILE_ANALYSIS_DEFAULT

            logger.info(
                f"Downloading attachment for analysis: {file_name} ({file_type}, ext={ext}) "
                f"using model {model} from {url[:80]}..."
            )

            # Set size limits (in bytes)
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB for PDFs
            MAX_TEXT_FILE_SIZE = (
                10 * 1024 * 1024
            )  # 10 MB for text-based files (CSV, Excel, DOCX, etc.)

            max_allowed_size = MAX_FILE_SIZE if is_pdf else MAX_TEXT_FILE_SIZE

            # Try to check file size from headers before downloading (HEAD request)
            try:
                head_resp = requests.head(url, allow_redirects=True, timeout=10)
                content_length = head_resp.headers.get("content-length")

                if content_length:
                    file_size = int(content_length)
                    if file_size > max_allowed_size:
                        raise ValueError(
                            f"File too large ({file_size / (1024 * 1024):.1f} MB). "
                            f"Maximum allowed size: {max_allowed_size / (1024 * 1024):.1f} MB"
                        )
            except Exception as e:
                # HEAD request failed or not supported, will check during download
                logger.debug(
                    f"HEAD request failed or not supported: {e}, will check size during download"
                )

            # Download file with streaming to check size incrementally
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code != 200:
                raise ValueError(
                    f"Failed to download attachment: HTTP {resp.status_code}"
                )

            # Also check content-length from GET response if available
            content_length = resp.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
                if file_size > max_allowed_size:
                    raise ValueError(
                        f"File too large ({file_size / (1024 * 1024):.1f} MB). "
                        f"Maximum allowed size: {max_allowed_size / (1024 * 1024):.1f} MB"
                    )

            # Download in chunks and check size incrementally
            file_bytes = b""
            chunk_size = 1024 * 1024  # 1 MB chunks
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    file_bytes += chunk
                    # Check size during download to avoid downloading entire large file
                    if len(file_bytes) > max_allowed_size:
                        raise ValueError(
                            f"File too large ({len(file_bytes) / (1024 * 1024):.1f} MB). "
                            f"Maximum allowed size: {max_allowed_size / (1024 * 1024):.1f} MB"
                        )
            # Determine file size for summary length guidance
            file_size_kb = len(file_bytes) / 1024
            summary_guidance = (
                "Provide a one-paragraph summary"
                if file_size_kb < 100
                else "Provide a two-paragraph summary"
            )

            # Build prompt based on whether user asked a specific question
            if user_question:
                prompt = (
                    "You are Lucy, a focused AI for digital advertising strategy and optimization. "
                    "Analyze this file and answer the user's specific question based on the file content. "
                    f"User's question: {user_question}\n\n"
                    f"{summary_guidance} that captures the key content, insights, and findings from this file. "
                    "Then provide a detailed answer to the user's question using information from the file. "
                    "Return a concise but useful analysis following the GenericFileAnalysis schema. "
                    "In the summary field, include both the file summary and your answer to the user's question. "
                    "Use simple, clear language suitable for non-expert marketers. "
                    "Be confident and specific in your answer based on what the file contains."
                )
            else:
                prompt = (
                    "You are Lucy, a focused AI for digital advertising strategy and optimization. "
                    "Your mission is to maximize ROAS with concise, actionable guidance. "
                    "Analyze this file and provide a comprehensive summary and marketing/advertising suggestions. "
                    f"{summary_guidance} that captures the key content, insights, and findings from this file. "
                    "Then provide generic marketing/advertising suggestions based on the file content, focusing on: "
                    "ROAS optimization opportunities, campaign strategy recommendations, audience targeting insights, "
                    "budget allocation suggestions, creative direction ideas, competitive positioning opportunities, "
                    "and actionable next steps that can be implemented in advertising platforms like Lofi. "
                    "Return a concise but useful analysis following the GenericFileAnalysis schema. "
                    "Use simple, clear language suitable for non-expert marketers. "
                    "Be confident and opinionated in what needs to be done and why."
                )

            if is_pdf:
                # PDF path: uses the OpenAI Files API to pass the document inline.
                # This is OpenAI-specific; other providers would need base64 inline content.
                # See image_provider.py for how to add alternative providers.
                client = get_openai_client()
                file_obj = io.BytesIO(file_bytes)
                file_obj.name = file_name

                uploaded = client.files.create(
                    file=file_obj,
                    purpose="assistants",
                )

                response = client.responses.parse(
                    model=model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": uploaded.id,
                                },
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    text_format=GenericFileAnalysis,
                )
                analysis: GenericFileAnalysis = response.output_parsed
            else:
                # Non-PDF path: render file content to text, then use provider-agnostic LLMClient
                if ext == ".csv":
                    rendered_text = _render_csv_for_llm(file_bytes)
                elif ext in {".xls", ".xlsx"}:
                    rendered_text = _render_excel_for_llm(file_bytes)
                elif ext in {".doc", ".docx"}:
                    rendered_text = _render_docx_for_llm(file_bytes)
                else:
                    rendered_text = _render_generic_text_for_llm(file_bytes)

                user_content = (
                    f"{prompt}\n\n"
                    f"File name: {file_name}\n"
                    f"File type: {file_type or ext or 'unknown'}\n\n"
                    f"File content starts below:\n\n"
                    f"{rendered_text}"
                )

                result = await _file_analysis_agent.run(
                    user_content,
                    model=to_responses_model(model),
                    model_settings=ModelSettings(temperature=0.0, max_tokens=1000),
                )
                analysis = result.output

            # Fill missing fields from metadata
            if not analysis.file_name:
                analysis.file_name = file_name
            if not analysis.file_type:
                analysis.file_type = ext.lstrip(".") if ext else file_type

            return {
                "success": True,
                "model_used": model,
                "file": {
                    "file_name": analysis.file_name,
                    "file_type": analysis.file_type,
                },
                "summary": analysis.summary,  # Explicitly include summary for easy access
                "analysis": analysis.model_dump(),
            }

        except Exception as e:
            logger.error(f"file_analysis_tool failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
