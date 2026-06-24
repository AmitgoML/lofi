"""FastAPI dependency accessors for objects built once in the app lifespan."""

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.supabase_client import SupabaseClient


def get_supabase_client(request: Request) -> SupabaseClient:
    return request.app.state.supabase_client


def get_bedrock_client(request: Request) -> BedrockClient:
    return request.app.state.bedrock_client


def get_compiled_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.compiled_graph


def get_workflow_errors(request: Request) -> dict[str, str]:
    return request.app.state.workflow_errors
