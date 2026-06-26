"""LangGraph wiring for the campaign planning workflow.

CampaignPlannerAgent is the orchestrator: it's the graph's entry point, and
every other agent hands control back to it. route_from_campaign_planner
decides what runs next purely from what's present in WorkflowState, so the
campaign_planner node itself only has to do the actual planning (and replan
on a QA FAIL) - it doesn't need to know about routing.

Two steps pause the workflow via LangGraph's interrupt()/Command(resume=...)
rather than by the router returning early: intake_form (missing intake
fields - see agents/lucy_intake.py) and human_review (final approve/reject -
see agents/human_review.py). Both require the compiled graph to be given a
checkpointer (see api/app.py); interrupt() raises without one.
"""

from langgraph.graph import END, StateGraph

from lofi.agents.campaign_planner import CampaignPlannerAgent
from lofi.agents.creative_director import CreativeDirectorAgent
from lofi.agents.human_review import HumanReviewAgent
from lofi.agents.lucy_intake import LucyCampaignIntake
from lofi.agents.performance_analyst import PerformanceAnalystAgent
from lofi.agents.qa_agent import QAAgent
from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.s3_storage import S3CreativeStorage
from lofi.persistence.supabase_client import SupabaseClient
from lofi.proposal.assembly import CampaignProposalAssembler
from lofi.state.workflow_state import WorkflowState


def route_from_campaign_planner(state: WorkflowState) -> str:
    """Picks the next hand-off from the orchestrator based on what's been
    produced so far. Unlike the old design, no pause-state special-casing is
    needed here: intake_form/human_review pause *inside* their own node via
    interrupt(), so the router only ever sees state from a node that has
    actually completed and returned.
    """
    if "intake_draft" not in state:
        return "intake_extract"
    if "campaign_brief" not in state:
        return "intake_form"
    if "performance_insights" not in state:
        return "performance_analyst"
    if "creative_director_output" not in state:
        return "creative_director"
    if "qa_result" not in state:
        return "qa_agent"
    if "campaign_proposal" not in state:
        return "proposal_assembly"
    if "approved" not in state:
        return "human_review"
    return END


def build_campaign_workflow_graph(
    supabase_client: SupabaseClient,
    bedrock_client: BedrockClient,
    s3_storage: S3CreativeStorage,
) -> StateGraph:
    intake = LucyCampaignIntake(bedrock_client)
    performance_analyst = PerformanceAnalystAgent(supabase_client)
    campaign_planner = CampaignPlannerAgent()
    creative_director = CreativeDirectorAgent(
        bedrock_client=bedrock_client,
        supabase_client=supabase_client,
        s3_storage=s3_storage,
    )
    qa_agent = QAAgent()
    proposal_assembler = CampaignProposalAssembler()
    human_review = HumanReviewAgent(supabase_client)

    graph = StateGraph(WorkflowState)

    graph.add_node("intake_extract", intake.extract)
    graph.add_node("intake_form", intake.collect_missing_fields)
    graph.add_node("performance_analyst", performance_analyst.run)
    graph.add_node("campaign_planner", campaign_planner.run)
    graph.add_node("creative_director", creative_director.run)
    graph.add_node("qa_agent", qa_agent.run)
    graph.add_node("proposal_assembly", proposal_assembler.run)
    graph.add_node("human_review", human_review.run)

    graph.set_entry_point("campaign_planner")

    # Every agent hands control back to the orchestrator rather than to the
    # next agent directly - campaign_planner is the only node that decides
    # what happens next.
    graph.add_edge("intake_extract", "campaign_planner")
    graph.add_edge("intake_form", "campaign_planner")
    graph.add_edge("performance_analyst", "campaign_planner")
    graph.add_edge("creative_director", "campaign_planner")
    graph.add_edge("qa_agent", "campaign_planner")
    graph.add_edge("proposal_assembly", "campaign_planner")
    graph.add_edge("human_review", "campaign_planner")

    graph.add_conditional_edges(
        "campaign_planner",
        route_from_campaign_planner,
        {
            END: END,
            "intake_extract": "intake_extract",
            "intake_form": "intake_form",
            "performance_analyst": "performance_analyst",
            "creative_director": "creative_director",
            "qa_agent": "qa_agent",
            "proposal_assembly": "proposal_assembly",
            "human_review": "human_review",
        },
    )

    return graph
