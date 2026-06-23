"""LangGraph wiring for the campaign planning workflow."""

from langgraph.graph import END, StateGraph

from lofi.agents.campaign_planner import CampaignPlannerAgent
from lofi.agents.creative_director import CreativeDirectorAgent
from lofi.agents.performance_analyst import PerformanceAnalystAgent
from lofi.agents.qa_agent import QAAgent
from lofi.intake.lucy_intake import LucyCampaignIntake
from lofi.proposal.assembly import CampaignProposalAssembler
from lofi.schemas.common import QAStatus
from lofi.state.workflow_state import WorkflowState


def route_after_intake(state: WorkflowState) -> str:
    """Pauses the workflow for form input when intake is missing required fields."""
    if state.get("intake_form_request") is not None:
        return END
    return "performance_analyst"


def route_after_qa(state: WorkflowState) -> str:
    """QA FAIL routes back to the Campaign Planner; PASS proceeds to assembly."""
    if state["qa_result"].status == QAStatus.FAIL:
        return "campaign_planner"
    return "proposal_assembly"


def build_campaign_workflow_graph() -> StateGraph:
    intake = LucyCampaignIntake()
    performance_analyst = PerformanceAnalystAgent()
    campaign_planner = CampaignPlannerAgent()
    creative_director = CreativeDirectorAgent()
    qa_agent = QAAgent()
    proposal_assembler = CampaignProposalAssembler()

    graph = StateGraph(WorkflowState)

    graph.add_node("intake", intake.run)
    graph.add_node("performance_analyst", performance_analyst.run)
    graph.add_node("campaign_planner", campaign_planner.run)
    graph.add_node("creative_director", creative_director.run)
    graph.add_node("qa_agent", qa_agent.run)
    graph.add_node("proposal_assembly", proposal_assembler.run)

    graph.set_entry_point("intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {END: END, "performance_analyst": "performance_analyst"},
    )
    graph.add_edge("performance_analyst", "campaign_planner")
    graph.add_edge("campaign_planner", "creative_director")
    graph.add_edge("creative_director", "qa_agent")
    graph.add_conditional_edges(
        "qa_agent",
        route_after_qa,
        {"campaign_planner": "campaign_planner", "proposal_assembly": "proposal_assembly"},
    )
    graph.add_edge("proposal_assembly", END)

    return graph
