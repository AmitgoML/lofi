# Lofi Campaign Planner

A multi-agent campaign planning workflow, served as a FastAPI app and orchestrated with LangGraph. A user describes a campaign in free text; a chain of agents turns that into a structured plan, creative assets, and a final proposal for human review before persisting to Supabase.

## Architecture

The Campaign Planner is the orchestrator: it's the graph's entry point, and every other agent (Intake, Performance Analyst, Creative Director, QA, Human Review) hands control back to it rather than to the next agent directly. A router (`route_from_campaign_planner` in `src/lofi/graph/workflow_graph.py`) decides what runs next purely from what's present in the shared `WorkflowState` (`src/lofi/state/workflow_state.py`) - it's the only place branching logic lives. QA `FAIL` triggers a replan, which clears the stale Creative Director/QA output so the router naturally sends the workflow through them again.

```
                 +---------------------+
        +------->|   campaign_planner   |<-------+
        |        +---------------------+        |
        |        |   |   |    |    |   |         |
        |        v   v   v    v    v   v         |
  intake_extract  |  perf_analyst   |   qa_agent  human_review
        |    intake_form    creative_director     |
        |         |               |          proposal_assembly
        +---------+---------------+----------------+
                          (all return to campaign_planner)
```

Run `uv run python scripts/visualize_graph.py` to regenerate an up-to-date diagram (`scripts/workflow_graph.png`) after changing the graph.

`intake_form` (missing intake fields) and `human_review` (final approve/reject) pause the workflow using LangGraph's `interrupt()`/`Command(resume=...)` rather than a hand-rolled status field - see `src/lofi/agents/lucy_intake.py` and `src/lofi/agents/human_review.py`. This requires the compiled graph to have a checkpointer (`MemorySaver`, wired in `src/lofi/api/app.py`), which is also what makes the pause durable across the separate HTTP requests that start/resume a workflow. **`MemorySaver` is in-process only - paused workflows are lost on server restart.** Swap it for a persistent checkpointer (e.g. `langgraph-checkpoint-postgres` against Supabase's Postgres connection) before production.

Extraction (the one Bedrock call in intake) lives in its own node, `intake_extract`, separate from `intake_form` (the node that actually pauses): LangGraph replays a node's whole function body on every resume, so anything with a side effect before an `interrupt()` *in the same node* would re-run on every resume too.

The same extraction call also classifies the request's *intent* (`campaign_planning` / `performance_analysis` / `creative_asset`, see `IntakeDraft.intent` in `src/lofi/schemas/intake.py`). `route_from_campaign_planner` branches on it: `campaign_planning` runs the full chain above; `performance_analysis` and `creative_asset` each run a single agent standalone (just enough intake to get a `brand`) and finish - no new nodes/edges, since those agents already hand control back to `campaign_planner` like everything else.

## Project layout

| Path | Purpose |
|---|---|
| `src/lofi/api/` | FastAPI app, routes, request/response schemas, dependency wiring |
| `src/lofi/agents/` | One module per agent in the pipeline above, including Lucy Campaign Intake (`agents/sub_agents/` for Copywriter/Image/Video) |
| `src/lofi/graph/` | LangGraph wiring (`workflow_graph.py`) |
| `src/lofi/state/` | `WorkflowState` TypedDict + `WorkflowStatus` (status is derived from the checkpointer, not stored) |
| `src/lofi/schemas/` | Pydantic input/output contracts for every agent |
| `src/lofi/llm/` | Bedrock client wrapper (Claude tool-use for structured extraction) |
| `src/lofi/persistence/` | Supabase client (metrics, campaigns, brand guidelines, creative assets) and S3 creative storage |
| `src/lofi/proposal/` | Final proposal assembly |
| `src/lofi/config/` | `Settings`, loaded from environment variables |

## Setup

```bash
uv sync
```

Required environment variables (see `src/lofi/config/settings.py`):

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Region for the Bedrock client |
| `BEDROCK_MODEL_ID` | Bedrock model ID used for intake extraction |
| `SUPABASE_URL` / `SUPABASE_KEY` | Supabase project credentials |
| `S3_BUCKET` | Bucket for generated creative assets |

## Running the API

```bash
uv run uvicorn lofi.main:app --reload --app-dir src
```

`--app-dir src` is needed because this project isn't installed as a package (`[tool.uv] package = false`) - it puts `src/` on `sys.path` so `lofi` is importable. pytest gets the same effect via `pythonpath = ["src"]` in `pyproject.toml`.

| Endpoint | Description |
|---|---|
| `POST /campaigns` | Start a workflow from a free-text request. Returns `202` with `workflow_id` immediately; the graph runs as a background task. |
| `GET /campaigns/{workflow_id}` | Poll status: `processing` / `awaiting_intake_form` / `awaiting_review` / `approved` / `rejected` / `failed`. Derived from `graph.get_state()` - whether the graph is paused on an `intake_form`/`human_review` interrupt, plus a small in-memory error map for `failed`. |
| `POST /campaigns/{workflow_id}/intake-form` | Resumes the paused `intake_form` interrupt with the submitted fields (`Command(resume=...)`), as a background task. May pause again if fields are still missing. |
| `POST /campaigns/{workflow_id}/approve` | Resumes the paused `human_review` interrupt with `{"approved": true}`; persists the campaign to Supabase synchronously. |
| `POST /campaigns/{workflow_id}/reject` | Resumes the paused `human_review` interrupt with `{"approved": false}`; no persistence. |

Starting a campaign and submitting an intake form run in the background (LLM calls, and eventually image/video generation, are too slow for a single request); approve/reject are synchronous since they only touch Supabase.

## Testing

```bash
uv run pytest
```

## Implementation status

Implemented: Performance Analyst (metrics aggregation/ranking), Lucy Campaign Intake (Bedrock-based extraction + interrupt-based form collection), Human Review (interrupt-based approve/reject + persistence), the FastAPI layer, and Supabase persistence for metrics/campaigns/brand guidelines/creative assets.

Still stubbed (raise `NotImplementedError`): `CampaignPlannerAgent.plan()`, `CreativeDirectorAgent.produce_assets()` and its sub-agents, `QAAgent.validate()`, and S3 creative storage. A full end-to-end `POST /campaigns` run will currently land on `status=failed` once it reaches the Campaign Planner (the exception is caught by the background task and surfaced via `GET /campaigns/{workflow_id}`, not raised to the server).

### Performance Analyst signals

Beyond the original ROAS/CTR/CPA ranking, `PerformanceAnalystAgent` (`src/lofi/agents/performance_analyst.py`) now computes, per recommendation:

- **`confidence`** (`high`/`medium`/`directional`) - how much historical evidence backs the number, based on sample size and (where a `cost` column is confirmed) total spend.
- **`anomalies`** - single days where a metric deviated more than 2 rolling standard deviations from its trailing 7-day mean.
- **`trend`** - week-over-week average change per metric.

All of the above is plain, deterministic Python (`confidence_label`/`compute_anomalies`/`compute_trend` near the top of the file) - no LLM involved, and fully unit-tested in `tests/agents/test_performance_signals.py`. Anomalies/trend only run for platforms and locations, because `campaign_platform_metrics`/`campaign_location_metrics` are the only tables with a confirmed `date` column (proven by the existing `.gte("date", ...)` filters in `SupabaseClient`); audience/creative get confidence only, since their date and `cost` columns aren't confirmed.

`PerformanceAnalystOutput.narrative_summary` is one optional LLM call (`BedrockClient.extract_structured`, same structured-extraction pattern used by Lucy Campaign Intake) that narrates the already-computed output in prose. It's intentionally the *only* place an LLM touches this agent, runs last, is constrained to the data already produced, and fails closed (`None`) on any error without affecting the rest of the output.

**Not yet wired into the production graph**: `build_campaign_workflow_graph` (`src/lofi/graph/workflow_graph.py`) still constructs `PerformanceAnalystAgent` without a `bedrock_client`, so `narrative_summary` is always `None` end-to-end today. Passing one in makes `extract_structured` get called on every workflow run, which conflicts with an existing invariant the intake pause/resume logic depends on (`test_routes.py::test_extraction_is_called_exactly_once_across_a_pause` asserts exactly one Bedrock call per workflow). Wire it up once that invariant is revisited.

Schema still considered too ambiguous to build on without confirming columns first: `campaign_id` and `budget_type` on the metrics/campaigns tables, and `campaign_status` - needed for per-campaign budget-reallocation signals (waste/saturation/headroom) and per-creative fatigue trend, neither of which exist yet.
