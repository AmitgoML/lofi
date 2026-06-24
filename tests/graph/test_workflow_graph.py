"""Unit tests for the campaign-planner-as-orchestrator router.

route_from_campaign_planner is a pure function of WorkflowState, so these
test it directly rather than running the compiled graph end-to-end. The
intake-form and human-review *pauses* themselves are no longer router
decisions (they're interrupt() calls inside their own nodes) - see
tests/api/test_routes.py for those, exercised through the real graph.
"""

from langgraph.graph import END

from lofi.graph.workflow_graph import route_from_campaign_planner


class TestRouteFromCampaignPlanner:
    def test_routes_to_intake_extract_when_no_draft_yet(self) -> None:
        assert route_from_campaign_planner({}) == "intake_extract"

    def test_routes_to_intake_form_once_draft_exists_but_no_brief(self) -> None:
        state = {"intake_draft": object()}

        assert route_from_campaign_planner(state) == "intake_form"

    def test_routes_to_performance_analyst_once_brief_is_ready(self) -> None:
        state = {"intake_draft": object(), "campaign_brief": object()}

        assert route_from_campaign_planner(state) == "performance_analyst"

    def test_routes_to_creative_director_once_insights_are_ready(self) -> None:
        state = {"intake_draft": object(), "campaign_brief": object(), "performance_insights": object()}

        assert route_from_campaign_planner(state) == "creative_director"

    def test_routes_to_qa_agent_once_creative_output_is_ready(self) -> None:
        state = {
            "intake_draft": object(),
            "campaign_brief": object(),
            "performance_insights": object(),
            "creative_director_output": object(),
        }

        assert route_from_campaign_planner(state) == "qa_agent"

    def test_routes_to_proposal_assembly_once_qa_result_is_ready(self) -> None:
        state = {
            "intake_draft": object(),
            "campaign_brief": object(),
            "performance_insights": object(),
            "creative_director_output": object(),
            "qa_result": object(),
        }

        assert route_from_campaign_planner(state) == "proposal_assembly"

    def test_routes_to_human_review_once_proposal_is_assembled(self) -> None:
        state = {
            "intake_draft": object(),
            "campaign_brief": object(),
            "performance_insights": object(),
            "creative_director_output": object(),
            "qa_result": object(),
            "campaign_proposal": object(),
        }

        assert route_from_campaign_planner(state) == "human_review"

    def test_routes_to_end_once_approved_or_rejected(self) -> None:
        state = {
            "intake_draft": object(),
            "campaign_brief": object(),
            "performance_insights": object(),
            "creative_director_output": object(),
            "qa_result": object(),
            "campaign_proposal": object(),
            "approved": False,
        }

        assert route_from_campaign_planner(state) == END

    def test_qa_fail_routes_back_to_creative_director_after_replan_clears_state(self) -> None:
        # campaign_planner.run() pops creative_director_output/qa_result on a
        # FAIL before the router ever sees this state, so from the router's
        # perspective this looks identical to "creative output not ready yet".
        state = {"intake_draft": object(), "campaign_brief": object(), "performance_insights": object()}

        assert route_from_campaign_planner(state) == "creative_director"
