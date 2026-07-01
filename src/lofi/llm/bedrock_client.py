"""Shared Bedrock client: extracts structured Pydantic objects from free text.

Uses Claude-on-Bedrock tool-use, forced to a single tool whose input_schema is
the target Pydantic model's JSON schema, so the response is guaranteed to be
shaped like that model rather than freeform text we'd have to parse ourselves.
"""

import json
from typing import Type, TypeVar

import boto3
from pydantic import BaseModel

from lofi.config.settings import Settings

ModelT = TypeVar("ModelT", bound=BaseModel)

ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 2048


class BedrockClient:
    """Thin wrapper around bedrock-runtime's invoke_model for structured extraction."""

    def __init__(self, settings: Settings, client=None) -> None:
        self._model_id = settings.bedrock_model_id
        if client is not None:
            self._client = client
        else:
            session = boto3.Session(
                profile_name=settings.aws_profile,
                region_name=settings.aws_region,
            )
            self._client = session.client("bedrock-runtime")

    def extract_structured(self, prompt: str, schema: Type[ModelT]) -> ModelT:
        tool_name = schema.__name__
        body = {
            "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": MAX_TOKENS,
            "tools": [
                {
                    "name": tool_name,
                    "description": f"Records the fields extracted from the request as a {tool_name}.",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{"role": "user", "content": prompt}],
        }

        response = self._client.invoke_model(modelId=self._model_id, body=json.dumps(body))
        payload = json.loads(response["body"].read())
        tool_use = next(block for block in payload["content"] if block["type"] == "tool_use")
        return schema.model_validate(tool_use["input"])

    def invoke_model(self, model_id: str, body: str) -> dict:
        """Raw invoke_model passthrough for non-Claude models (e.g. Nova Canvas)."""
        response = self._client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())
