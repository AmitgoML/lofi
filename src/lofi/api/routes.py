"""HTTP endpoints for the campaign planning workflow.

Starting a campaign and submitting an intake form both kick the LangGraph
workflow off as a background task (LLM calls and, eventually, image/video
generation make a synchronous request too slow) and return immediately;
clients poll GET /campaigns/{workflow_id} for progress. Approve/reject are
synchronous since they only touch Supabase, not agents.

The intake-form pause and the final human review both use LangGraph's
interrupt()/Command(resume=...) (see agents/lucy_intake.py and
agents/human_review.py) rather than a hand-rolled "status" field stored
alongside the state - the checkpointer (MemorySaver, see api/app.py) is the
source of truth for what's paused and why, read here via graph.get_state().
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot

from lofi.api.dependencies import get_compiled_graph, get_workflow_errors
from lofi.api.schemas import ApprovalResponse, CampaignStatusResponse, StartCampaignRequest, WorkflowResponse
from lofi.schemas.intake import IntakeDraft, IntakeFormRequest
from lofi.state.workflow_state import WorkflowState, WorkflowStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _thread_config(workflow_id: str) -> dict:
    return {"configurable": {"thread_id": workflow_id}}


def _run_graph(
    graph: CompiledStateGraph, workflow_id: str, input_: "WorkflowState | Command", errors: dict[str, str]
) -> None:
    try:
        graph.invoke(input_, config=_thread_config(workflow_id))
    except Exception as exc:  # noqa: BLE001 - any agent/LLM/Supabase failure should land here as a status, not crash the task
        # str(exc) can be "" for bare `raise SomeError` with no message (e.g.
        # the still-unimplemented agent stubs) - log the full traceback so
        # the failure is visible somewhere even when the API response isn't.
        logger.exception("Workflow %s failed", workflow_id)
        errors[workflow_id] = str(exc) or repr(exc)


def _pending_interrupt(snapshot: StateSnapshot) -> Interrupt | None:
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


def _get_snapshot_or_404(workflow_id: str, graph: CompiledStateGraph) -> StateSnapshot:
    snapshot = graph.get_state(_thread_config(workflow_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return snapshot


@router.post("/campaigns", response_model=WorkflowResponse, status_code=202)
def start_campaign(
    request: StartCampaignRequest,
    background_tasks: BackgroundTasks,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    errors: dict[str, str] = Depends(get_workflow_errors),
) -> WorkflowResponse:
    workflow_id = str(uuid4())
    initial_state: WorkflowState = {
        "user_request": request.user_request,
        "organization_id": request.organization_id,
        "organization_max_budget": request.organization_max_budget,
    }
    background_tasks.add_task(_run_graph, graph, workflow_id, initial_state, errors)
    return WorkflowResponse(workflow_id=workflow_id, status=WorkflowStatus.PROCESSING)


@router.get("/campaigns/{workflow_id}", response_model=CampaignStatusResponse)
def get_campaign_status(
    workflow_id: str,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    errors: dict[str, str] = Depends(get_workflow_errors),
) -> CampaignStatusResponse:
    snapshot = _get_snapshot_or_404(workflow_id, graph)
    intake_draft = snapshot.values.get("intake_draft")
    intent = intake_draft.intent if intake_draft is not None else None

    if workflow_id in errors:
        return CampaignStatusResponse(
            workflow_id=workflow_id, status=WorkflowStatus.FAILED, intent=intent, error=errors[workflow_id]
        )

    pending = _pending_interrupt(snapshot)

    intake_form_request = None
    if pending is not None and pending.value.get("type") == "intake_form":
        status = WorkflowStatus.AWAITING_INTAKE_FORM
        intake_form_request = IntakeFormRequest(missing_fields=pending.value["missing_fields"])
    elif pending is not None and pending.value.get("type") == "human_review":
        status = WorkflowStatus.AWAITING_REVIEW
    elif snapshot.values.get("approved") is True:
        status = WorkflowStatus.APPROVED
    elif snapshot.values.get("approved") is False:
        status = WorkflowStatus.REJECTED
    elif not snapshot.next and (
        "performance_insights" in snapshot.values or "creative_director_output" in snapshot.values
    ):
        # performance_analysis/creative_asset intents finish here directly -
        # they never go through human_review, so "approved" is never set for
        # them. `not snapshot.next` confirms the graph actually reached END
        # rather than just being mid-way through the campaign_planning chain.
        status = WorkflowStatus.COMPLETED
    else:
        status = WorkflowStatus.PROCESSING

    return CampaignStatusResponse(
        workflow_id=workflow_id,
        status=status,
        intent=intent,
        intake_form_request=intake_form_request,
        performance_insights=snapshot.values.get("performance_insights"),
        creative_director_output=snapshot.values.get("creative_director_output"),
        campaign_proposal=snapshot.values.get("campaign_proposal"),
    )


@router.post("/campaigns/{workflow_id}/intake-form", response_model=WorkflowResponse, status_code=202)
def submit_intake_form(
    workflow_id: str,
    submission: IntakeDraft,
    background_tasks: BackgroundTasks,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    errors: dict[str, str] = Depends(get_workflow_errors),
) -> WorkflowResponse:
    snapshot = _get_snapshot_or_404(workflow_id, graph)
    pending = _pending_interrupt(snapshot)
    if pending is None or pending.value.get("type") != "intake_form":
        raise HTTPException(status_code=409, detail="Workflow is not awaiting an intake form")

    background_tasks.add_task(
        _run_graph, graph, workflow_id, Command(resume=submission.model_dump(exclude_none=True)), errors
    )
    return WorkflowResponse(workflow_id=workflow_id, status=WorkflowStatus.PROCESSING)


@router.post("/campaigns/{workflow_id}/approve", response_model=ApprovalResponse)
def approve_campaign(
    workflow_id: str,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    errors: dict[str, str] = Depends(get_workflow_errors),
) -> ApprovalResponse:
    snapshot = _get_snapshot_or_404(workflow_id, graph)
    pending = _pending_interrupt(snapshot)
    if pending is None or pending.value.get("type") != "human_review":
        raise HTTPException(status_code=409, detail="Workflow is not awaiting review")

    _run_graph(graph, workflow_id, Command(resume={"approved": True}), errors)
    if workflow_id in errors:
        raise HTTPException(status_code=500, detail=errors[workflow_id])

    result = graph.get_state(_thread_config(workflow_id))
    return ApprovalResponse(
        workflow_id=workflow_id,
        status=WorkflowStatus.APPROVED,
        persisted_campaign_id=result.values.get("persisted_campaign_id"),
    )


@router.post("/campaigns/{workflow_id}/reject", response_model=ApprovalResponse)
def reject_campaign(
    workflow_id: str,
    graph: CompiledStateGraph = Depends(get_compiled_graph),
    errors: dict[str, str] = Depends(get_workflow_errors),
) -> ApprovalResponse:
    snapshot = _get_snapshot_or_404(workflow_id, graph)
    pending = _pending_interrupt(snapshot)
    if pending is None or pending.value.get("type") != "human_review":
        raise HTTPException(status_code=409, detail="Workflow is not awaiting review")

    _run_graph(graph, workflow_id, Command(resume={"approved": False}), errors)
    if workflow_id in errors:
        raise HTTPException(status_code=500, detail=errors[workflow_id])

    return ApprovalResponse(workflow_id=workflow_id, status=WorkflowStatus.REJECTED)
