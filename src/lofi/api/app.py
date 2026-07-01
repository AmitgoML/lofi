"""FastAPI app factory: wires Settings/SupabaseClient/BedrockClient/graph
once at startup (lifespan) rather than per-request."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from lofi.api.routes import router
from lofi.config.settings import Settings
from lofi.graph.workflow_graph import build_campaign_workflow_graph
from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.s3_storage import S3CreativeStorage
from lofi.persistence.supabase_client import SupabaseClient

# Settings() reads .env itself, but boto3 (BedrockClient) reads
# AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY straight from the process
# environment - load_dotenv() exports the whole file there too so both pick
# it up without needing the shell session to export them first.
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    supabase_client = SupabaseClient(settings)
    bedrock_client = BedrockClient(settings)
    s3_storage = S3CreativeStorage(settings)

    # MemorySaver checkpoints interrupt()-paused workflows (intake form,
    # human review) in-process only - state is lost on restart and unsafe
    # across multiple server instances. Swap for a persistent checkpointer
    # (e.g. langgraph-checkpoint-postgres against Supabase's Postgres
    # connection) before production.
    checkpointer = MemorySaver()

    app.state.supabase_client = supabase_client
    app.state.bedrock_client = bedrock_client
    app.state.s3_storage = s3_storage
    app.state.compiled_graph = build_campaign_workflow_graph(
        supabase_client, bedrock_client, s3_storage
    ).compile(checkpointer=checkpointer)
    # Background graph runs can fail (agent/LLM/Supabase errors); since
    # there's no "FAILED" concept in LangGraph's own checkpoint state, the
    # background task records it here for GET /campaigns/{id} to surface.
    app.state.workflow_errors = {}
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Lofi Campaign Planner", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
