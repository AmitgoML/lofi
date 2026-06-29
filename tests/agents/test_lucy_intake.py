"""Unit tests for LucyCampaignIntake.

extract() is tested directly (no interrupt involved). collect_missing_fields()
calls interrupt(), which raises if invoked outside a running LangGraph graph -
so it's tested through a minimal single-node graph with a checkpointer here,
and through the full app in tests/api/test_routes.py.
"""

from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from lofi.agents.lucy_intake import EXTRACTION_PROMPT_TEMPLATE, LucyCampaignIntake
from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, Location, Platform
from lofi.schemas.intake import ExtractedIntakeFields, Intent, IntakeDraft, IntakeField
from lofi.state.workflow_state import WorkflowState


@pytest.fixture
def bedrock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def intake(bedrock_client: MagicMock) -> LucyCampaignIntake:
    return LucyCampaignIntake(bedrock_client)


class TestExtractBrief:
    def test_calls_bedrock_with_extraction_prompt_and_schema(
        self, intake: LucyCampaignIntake, bedrock_client: MagicMock
    ) -> None:
        bedrock_client.extract_structured.return_value = ExtractedIntakeFields()

        intake.extract_brief("Promote my coffee shop")

        bedrock_client.extract_structured.assert_called_once_with(
            EXTRACTION_PROMPT_TEMPLATE.format(user_request="Promote my coffee shop"), ExtractedIntakeFields
        )

    def test_unmentioned_fields_stay_none(self, intake: LucyCampaignIntake, bedrock_client: MagicMock) -> None:
        bedrock_client.extract_structured.return_value = ExtractedIntakeFields(brand="Acme Coffee")

        draft = intake.extract_brief("Promote my coffee shop")

        assert draft.locations is None
        assert draft.target_audience is None
        assert draft.platforms is None
        assert draft.campaign_timing is None

    def test_unclassified_intent_falls_back_to_campaign_planning(
        self, intake: LucyCampaignIntake, bedrock_client: MagicMock
    ) -> None:
        bedrock_client.extract_structured.return_value = ExtractedIntakeFields()

        draft = intake.extract_brief("Promote my coffee shop")

        assert draft.intent == Intent.CAMPAIGN_PLANNING

    def test_classified_intent_is_kept(self, intake: LucyCampaignIntake, bedrock_client: MagicMock) -> None:
        bedrock_client.extract_structured.return_value = ExtractedIntakeFields(intent=Intent.PERFORMANCE_ANALYSIS)

        draft = intake.extract_brief("How did our Meta ads perform last month?")

        assert draft.intent == Intent.PERFORMANCE_ANALYSIS


class TestExtractNode:
    def test_extracts_and_injects_organization_id(self, intake: LucyCampaignIntake, bedrock_client: MagicMock) -> None:
        bedrock_client.extract_structured.return_value = ExtractedIntakeFields(brand="Acme Coffee")
        state: WorkflowState = {"user_request": "Promote my coffee shop", "organization_id": "org-1"}

        result_state = intake.extract(state)

        assert result_state["intake_draft"].brand == "Acme Coffee"
        assert result_state["intake_draft"].organization_id == "org-1"

    def test_does_not_re_extract_once_draft_exists(self, intake: LucyCampaignIntake, bedrock_client: MagicMock) -> None:
        existing_draft = IntakeDraft(user_request="Promote my coffee shop", organization_id="org-1", brand="Acme Coffee")
        state: WorkflowState = {
            "user_request": "Promote my coffee shop",
            "organization_id": "org-1",
            "intake_draft": existing_draft,
        }

        intake.extract(state)

        bedrock_client.extract_structured.assert_not_called()


class TestFindMissingFields:
    def test_lists_every_unset_field_for_campaign_planning(self, intake: LucyCampaignIntake) -> None:
        draft = IntakeDraft(user_request="x", organization_id="org-1", brand="Acme Coffee")

        missing = intake.find_missing_fields(draft)

        assert IntakeField.BRAND not in missing
        assert IntakeField.GOAL in missing

    def test_performance_analysis_only_requires_brand(self, intake: LucyCampaignIntake) -> None:
        draft = IntakeDraft(user_request="x", organization_id="org-1", intent=Intent.PERFORMANCE_ANALYSIS)

        assert intake.find_missing_fields(draft) == [IntakeField.BRAND]

        draft = draft.model_copy(update={"brand": "Acme Coffee"})
        assert intake.find_missing_fields(draft) == []

    def test_creative_asset_only_requires_brand(self, intake: LucyCampaignIntake) -> None:
        draft = IntakeDraft(user_request="x", organization_id="org-1", intent=Intent.CREATIVE_ASSET)

        assert intake.find_missing_fields(draft) == [IntakeField.BRAND]


class TestApplyFormSubmission:
    def test_merges_non_none_fields_into_draft(self, intake: LucyCampaignIntake) -> None:
        draft = IntakeDraft(user_request="Promote my coffee shop", organization_id="org-1")
        submission = IntakeDraft(user_request="ignored", brand="Acme Coffee")

        merged = intake.apply_form_submission(draft, submission)

        assert merged.brand == "Acme Coffee"
        assert merged.user_request == "Promote my coffee shop"
        assert merged.organization_id == "org-1"

    def test_nested_fields_are_revalidated_not_left_as_dicts(self, intake: LucyCampaignIntake) -> None:
        draft = IntakeDraft(user_request="Promote my coffee shop", organization_id="org-1")
        submission = IntakeDraft(user_request="ignored", budget=BudgetSpec(total_budget=500.0))

        merged = intake.apply_form_submission(draft, submission)

        assert isinstance(merged.budget, BudgetSpec)
        assert merged.budget.total_budget == 500.0


class TestCollectMissingFields:
    """Exercises the interrupt()/Command(resume=...) loop through a minimal
    single-node graph, since interrupt() requires a running graph."""

    def _build_graph(self, intake: LucyCampaignIntake):
        graph = StateGraph(WorkflowState)
        graph.add_node("collect", intake.collect_missing_fields)
        graph.set_entry_point("collect")
        graph.add_edge("collect", END)
        return graph.compile(checkpointer=MemorySaver())

    def test_pauses_with_missing_fields_then_resumes_to_completion(self, intake: LucyCampaignIntake) -> None:
        compiled = self._build_graph(intake)
        config = {"configurable": {"thread_id": "t1"}}
        draft = IntakeDraft(user_request="Promote my coffee shop", organization_id="org-1")

        first = compiled.invoke({"intake_draft": draft}, config=config)
        assert "__interrupt__" in first
        interrupt_payload = first["__interrupt__"][0].value
        assert interrupt_payload["type"] == "intake_form"
        assert "brand" in interrupt_payload["missing_fields"]

        second = compiled.invoke(
            Command(
                resume={
                    "user_request": "ignored",
                    "brand": "Acme Coffee",
                    "goal": "awareness",
                    "budget": {"total_budget": 500.0},
                    "campaign_timing": {"start_date": "2026-07-01"},
                    "locations": [{"country": "USA"}],
                    "target_audience": {"age_min": 18, "age_max": 35},
                    "platforms": ["meta"],
                }
            ),
            config=config,
        )

        assert "__interrupt__" not in second
        assert second["campaign_brief"].brand == "Acme Coffee"

    def test_multiple_rounds_of_missing_fields(self, intake: LucyCampaignIntake) -> None:
        compiled = self._build_graph(intake)
        config = {"configurable": {"thread_id": "t2"}}
        draft = IntakeDraft(user_request="Promote my coffee shop", organization_id="org-1")

        compiled.invoke({"intake_draft": draft}, config=config)
        second = compiled.invoke(Command(resume={"user_request": "ignored", "brand": "Acme Coffee"}), config=config)

        assert "__interrupt__" in second
        assert "brand" not in second["__interrupt__"][0].value["missing_fields"]
        assert "goal" in second["__interrupt__"][0].value["missing_fields"]

        third = compiled.invoke(
            Command(
                resume={
                    "user_request": "ignored",
                    "goal": "awareness",
                    "budget": {"total_budget": 500.0},
                    "campaign_timing": {"start_date": "2026-07-01"},
                    "locations": [{"country": "USA"}],
                    "target_audience": {"age_min": 18, "age_max": 35},
                    "platforms": ["meta"],
                }
            ),
            config=config,
        )

        assert "__interrupt__" not in third
        assert third["campaign_brief"].goal == CampaignGoal.AWARENESS
        assert third["campaign_brief"].platforms == [Platform.META]
        assert third["campaign_brief"].target_audience == AudienceSpec(age_min=18, age_max=35)
        assert third["campaign_brief"].locations == [Location(country="USA")]

    def test_performance_analysis_intent_skips_campaign_brief(self, intake: LucyCampaignIntake) -> None:
        compiled = self._build_graph(intake)
        config = {"configurable": {"thread_id": "t3"}}
        draft = IntakeDraft(
            user_request="How did our ads do?", organization_id="org-1", intent=Intent.PERFORMANCE_ANALYSIS
        )

        first = compiled.invoke({"intake_draft": draft}, config=config)
        assert "__interrupt__" in first
        assert first["__interrupt__"][0].value["missing_fields"] == ["brand"]

        second = compiled.invoke(Command(resume={"user_request": "ignored", "brand": "Acme Coffee"}), config=config)

        assert "__interrupt__" not in second
        assert second["intake_draft"].brand == "Acme Coffee"
        assert "campaign_brief" not in second
