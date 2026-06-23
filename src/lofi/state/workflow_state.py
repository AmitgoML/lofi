"""Shared workflow state passed between LangGraph nodes.

Field types are the Pydantic schemas in :mod:`lofi.schemas`, which are the
canonical input/output contracts for each agent.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.campaign_planner import CampaignPlannerInput, FinalCampaignProposal
from lofi.schemas.creative_director import CreativeBrief, CreativeDirectorOutput
from lofi.schemas.intake import IntakeDraft, IntakeFormRequest
from lofi.schemas.performance_analyst import PerformanceAnalystOutput
from lofi.schemas.qa_agent import QAAgentOutput


class WorkflowState(TypedDict, total=False):
    """LangGraph graph state for the campaign planning workflow."""

    user_request: str
    organization_max_budget: float
    intake_draft: IntakeDraft
    intake_form_request: Optional[IntakeFormRequest]
    campaign_brief: CampaignPlannerInput
    performance_insights: PerformanceAnalystOutput
    campaign_plan: CampaignPlan
    creative_brief: CreativeBrief
    creative_director_output: CreativeDirectorOutput
    qa_result: QAAgentOutput
    campaign_proposal: Optional[FinalCampaignProposal]
