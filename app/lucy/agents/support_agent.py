import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.base_agent import LofiAgent
from lucy.agents.common.model_config import Models, to_responses_model
from lucy.agents.common.models import ChatDeps, FileAgentOutput
from lucy.agents.common.navigation import LOFI_NAVIGATION_BLOCK
from lucy.agents.common.tools import register_user_org_profiles_tool
from lucy.kb.knowledge_base import get_lofi_knowledge_base

SUPPORT_SYSTEM_PROMPT = """
You are **Lucy**, Lofi's in-app Support Agent. Be concise, friendly, and action-oriented.

### Tool Usage (strict)
- Use `lofi_guide_tool` exclusively to retrieve product knowledge from the Lofi KB (docs, FAQs, troubleshooting, policies).
- Do not call any other tools except `get_user_org_profiles_tool`. Do not use web search.
- The tool automatically enhances searches with user position context when available for more relevant results.

- Summarize KB answers in your own words. When it adds credibility, include the doc title/section in-line.

### Operating Principles (Always)
1) **Clarify → Goal**: If the user's intent is unclear, ask up to **two** pointed questions to define the goal and "what success looks like."
2) **Ground → Knowledge Base**: Search the internal KB first (product docs, FAQs, troubleshooting, mirrored policies). Prefer the most relevant, most recent entries. Summarize in your words and **cite the doc title/section**.
3) **Use → Context**: Adapt to the current **screen** and **active object**. Consider user **role**, brand guardrails, plan limits, and campaign status.
4) **Offer → Actions**: If an operation helps (pause/resume, run compliance, update billing, invite user), propose it. **Ask for confirmation** before executing unless the user explicitly commanded it.
5) **Truthfulness**: If you don't know or the capability is unsupported, say so and provide next steps or a safe workaround.
6) **Escalate → Ticket**: If blocked, offer to **submit a support ticket**. Include screen, object IDs, last steps, and user context. If the user agrees, call the ticket tool.
7) **Compliance**: Before launch-critical advice, recommend or run a **pre-launch compliance check**.
8) **RBAC & Safety**: Respect permissions. Don't expose internal links or private data. Don't promise non-GA features. Avoid hallucinations (use the capabilities catalog).
9) **Style**: Prefer numbered steps, short checklists, and clear next actions. Keep answers crisp. Never use emojis or emoticons under any circumstances — including when apologizing or expressing empathy. Plain professional text only.

### If KB doesn't resolve it or capability seems unsupported
- Offer at most **one repeat of the same KB-derived solution**.
- If still unresolved, or KB results are missing/unclear, **check the conversation history** for the latest question or context that might provide additional clues.
- **Always prioritize helping with the latest question** - if the user asked about creating a campaign, help them create a campaign. If they asked about specific features, help with those features.
- If conversation history doesn't help, acknowledge limitation ("It looks like I don't have a complete answer for this").
- State that the functionality is probably unsupported in Lofi. Do not guess.
 - Invite the user to request the capability in Lofi and offer to forward it to the product team; provide support@meetlofi.com to initiate the feature request.
- Offer to open a support ticket at support@meetlofi.com or handoff. If a safe workaround exists, include it.
- Never re-state the same steps more than twice.

### KB Navigation Order (when searching)
1) Troubleshooting & FAQs (quick resolutions)
2) Product docs (end-to-end flows, feature behavior)
3) Policies (platform, industry) – mirrored and versioned internally
If conflicting, prefer the newest internal doc. Disclose ambiguity if unresolved.

### Follow-up
End with one short follow-up question that advances the user's current goal.
The question should be informed by what the user is trying to accomplish — not a generic upsell.
Mention Lofi only when the natural next step involves a Lofi feature (e.g. creating a campaign, launching an ad).
Never conflate Lofi (the platform) with the user's brand or company.

Skip greetings/small talk.
""" + LOFI_NAVIGATION_BLOCK


class SupportAgent(LofiAgent):
    """Factory and metadata for the Support agent."""

    REMINDER_HEADER = (
        "You are Lucy Support — concise, friendly, action-oriented. "
        "Use only lofi_guide_tool (no web search). Prioritize the user's latest question. "
        "Call tools silently — no announcement text before a tool call. "
        "End with one contextual follow-up question."
    )

    @classmethod
    def create(cls, model_name: Optional[str] = None) -> Agent:
        model_name = to_responses_model(model_name or Models.SUPPORT)
        logger.info(f"Creating support agent with model '{model_name}'")
        agent = Agent(
            model=model_name,
            model_settings=ModelSettings(temperature=0.2, max_tokens=1500),
            deps_type=ChatDeps,
            system_prompt=SUPPORT_SYSTEM_PROMPT,
            output_type=FileAgentOutput,
        )
        register_user_org_profiles_tool(agent)

        @agent.tool
        async def lofi_guide_tool(
            ctx: RunContext[ChatDeps],
            question: str = Field(
                description="User question about Lofi or using the platform"
            ),
            top_k: int = Field(
                default=3, description="Number of Lofi KB results to use (1-5)"
            ),
        ) -> Dict[str, Any]:
            """
            Search the Lofi product knowledge base for guidance and format concrete in-app steps
            with exact routes. Returns a structured payload with synthesized guidance and steps.
            """
            ctx.deps.status_queue.put_nowait("Searching Lofi guides")
            logger.info(f"Lofi guide requested: '{question}' (top_k={top_k})")

            # Clamp top_k
            if top_k < 1:
                top_k = 1
            elif top_k > 5:
                top_k = 5

            # Enhance search query with user position if available
            enhanced_question = question
            if ctx.deps.user_location:
                enhanced_question = (
                    f"{question} (user position: {ctx.deps.user_location})"
                )
                logger.info(
                    f"Enhanced search query with position: '{enhanced_question}'"
                )

            # Search Lofi KB (RAG)
            try:
                timeout_s = float(os.getenv("SEMANTIC_SEARCH_TIMEOUT_SEC", "5.0"))
                lofi_kb = get_lofi_knowledge_base()
                results = await asyncio.wait_for(
                    asyncio.to_thread(lofi_kb.search, enhanced_question, top_k),
                    timeout=timeout_s,
                )
            except Exception as e:
                logger.error(f"Error searching Lofi KB: {e}")
                results = []

            guidance_parts: List[str] = []
            steps: List[str] = []
            sources: List[str] = []

            for item, score in results:
                if item.answer:
                    guidance_parts.append(item.answer)
                if item.implications:
                    steps.extend(item.implications)
                if item.source:
                    sources.append(str(item.source))

            def dedupe(seq: List[str]) -> List[str]:
                seen: set[str] = set()
                out: List[str] = []
                for s in seq:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                return out

            guidance_parts = dedupe(
                [g.strip() for g in guidance_parts if g and g.strip()]
            )
            sources = dedupe([s.strip() for s in sources if s and s.strip()])
            steps = dedupe([s.strip() for s in steps if s and s.strip()])

            formatted_steps: List[str] = []
            extracted_routes: List[str] = []
            for step in steps:
                s = step
                ticks = re.findall(r"`([^`]+)`", s) or []
                if ticks:
                    for r in ticks:
                        extracted_routes.append(r)
                formatted_steps.append(s)

            if not sources:
                sources = ["Lofi Knowledge Base (`help`)"]

            payload: Dict[str, Any] = {
                "guidance": "\n\n".join(guidance_parts) if guidance_parts else "",
                "steps": formatted_steps,
                "sources": sources[:3],
                "routes": dedupe(extracted_routes),
            }

            logger.info(
                f"Lofi guide built with {len(guidance_parts)} guidance parts, {len(formatted_steps)} steps"
            )
            return payload

        return agent
