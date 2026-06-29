"""Shared workflow state passed between LangGraph nodes.

Field types are the Pydantic schemas in :mod:`lofi.schemas`, which are the
canonical input/output contracts for each agent.

No status/error/intake_form_request fields live here anymore: the
checkpointer (see api/app.py) is the source of truth for what's paused and
why, and the API layer (api/routes.py) derives WorkflowStatus from
graph.get_state() rather than from a field stored in this dict.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, TypedDict

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import CampaignPlannerInput, FinalCampaignProposal
from lofi.schemas.creative_director import CreativeBrief, CreativeDirectorOutput
from lofi.schemas.intake import IntakeDraft
from lofi.schemas.performance_analyst import PerformanceAnalystOutput
from lofi.schemas.qa_agent import QAAgentOutput


class WorkflowStatus(str, Enum):
    """The API layer's view of workflow progress (derived, not stored)."""

    PROCESSING = "processing"
    AWAITING_INTAKE_FORM = "awaiting_intake_form"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    # performance_analysis/creative_asset intents (see Intent in
    # schemas/intake.py) finish without ever going through human_review, so
    # they never set "approved" - COMPLETED is their terminal status instead.
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowState(TypedDict, total=False):
    """LangGraph graph state for the campaign planning workflow."""

    user_request: str
    organization_id: str
    organization_max_budget: float
    intake_draft: IntakeDraft
    campaign_brief: CampaignPlannerInput
    performance_insights: PerformanceAnalystOutput
    campaign_plan: CampaignPlan
    creative_brief: CreativeBrief
    creative_director_output: CreativeDirectorOutput
    qa_result: QAAgentOutput
    campaign_proposal: FinalCampaignProposal
    approved: bool
    persisted_campaign_id: Optional[str]
