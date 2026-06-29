"""Integration tests for the campaign workflow API.

These exercise the real compiled LangGraph (with a MemorySaver checkpointer)
through interrupt()/Command(resume=...) - not a mocked-out graph - since
that's the behavior actually being tested. SupabaseClient/BedrockClient are
monkeypatched at the method level (no real network calls), and the three
agent methods that are still implementation stubs (CampaignPlannerAgent.plan,
CreativeDirectorAgent.produce_assets, QAAgent.validate) are monkeypatched to
canned outputs so a full run can reach the human_review interrupt.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from lofi.agents.campaign_planner import CampaignPlannerAgent
from lofi.agents.creative_director import CreativeDirectorAgent
from lofi.agents.qa_agent import QAAgent
from lofi.api.app import create_app
from lofi.llm.bedrock_client import BedrockClient
from lofi.persistence.supabase_client import SupabaseClient
from lofi.schemas.campaign_plan import CampaignPlan
from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, Location, Platform, QAStatus
from lofi.schemas.creative_director import CreativeBrief, CreativeDirectorOutput, TextAsset
from lofi.schemas.intake import ExtractedIntakeFields, Intent
from lofi.schemas.performance_analyst import NarrativeSummary
from lofi.schemas.qa_agent import QAAgentOutput
from lofi.state.workflow_state import WorkflowStatus

SAMPLE_PLAN = CampaignPlan(
    goal=CampaignGoal.AWARENESS,
    campaign_type="launch",
    objective="reach",
    audience=AudienceSpec(age_min=18, age_max=35),
    platforms=[Platform.META],
    locations=[Location(country="USA")],
    budget=BudgetSpec(total_budget=500.0),
    timing=CampaignTiming(start_date="2026-07-01"),
)
SAMPLE_CREATIVE_BRIEF = CreativeBrief(
    goal=CampaignGoal.AWARENESS, audience=AudienceSpec(age_min=18, age_max=35), platforms=[Platform.META]
)
SAMPLE_CREATIVE_OUTPUT = CreativeDirectorOutput(
    asset_decisions=[],
    best_creative_format="text",
    best_messaging_angle="value for money",
    assets=[],
    texts=TextAsset(headlines=["Wake up to Acme"], descriptions=["Best coffee in town"], cta="Order now"),
)
SAMPLE_QA_PASS = QAAgentOutput(
    status=QAStatus.PASS,
    budget_validation_passed=True,
    platform_compatibility_passed=True,
    creative_completeness_passed=True,
    required_fields_passed=True,
    policy_compliance_passed=True,
)

FULLY_SPECIFIED_FIELDS = ExtractedIntakeFields(
    brand="Acme Coffee",
    goal=CampaignGoal.AWARENESS,
    budget=BudgetSpec(total_budget=500.0),
    campaign_timing=CampaignTiming(start_date="2026-07-01"),
    locations=[Location(country="USA")],
    target_audience=AudienceSpec(age_min=18, age_max=35),
    platforms=[Platform.META],
)


@pytest.fixture(autouse=True)
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-3")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")


@pytest.fixture(autouse=True)
def stub_supabase_methods(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    save_campaign = MagicMock(return_value="campaign-123")
    monkeypatch.setattr(SupabaseClient, "get_platform_metrics", lambda self, *a, **kw: [])
    monkeypatch.setattr(SupabaseClient, "get_location_metrics", lambda self, *a, **kw: [])
    monkeypatch.setattr(SupabaseClient, "get_audience_metrics", lambda self, *a, **kw: [])
    monkeypatch.setattr(SupabaseClient, "get_creative_metrics", lambda self, *a, **kw: [])
    monkeypatch.setattr(SupabaseClient, "save_campaign", save_campaign)
    return save_campaign


def _stub_creative_director_run(self: CreativeDirectorAgent, state: dict) -> dict:
    # CreativeDirectorAgent.run() is still an unimplemented stub (raises
    # immediately) - patch the node itself rather than produce_assets so the
    # rest of the pipeline can be exercised here.
    state["creative_director_output"] = SAMPLE_CREATIVE_OUTPUT
    return state


@pytest.fixture(autouse=True)
def stub_agent_implementations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CampaignPlannerAgent, "plan", lambda self, campaign_brief, performance_insights: (SAMPLE_PLAN, SAMPLE_CREATIVE_BRIEF)
    )
    monkeypatch.setattr(CreativeDirectorAgent, "run", _stub_creative_director_run)
    monkeypatch.setattr(QAAgent, "validate", lambda self, qa_input: SAMPLE_QA_PASS)


@pytest.fixture
def extract_structured(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    # Shared across two callers now: intake extraction (schema=ExtractedIntakeFields,
    # default-mocked below to FULLY_SPECIFIED_FIELDS, overridden per-test) and the
    # performance analyst's narrative summary (schema=NarrativeSummary). Branch on
    # the schema arg so each gets a response of the right type.
    def _respond(prompt: str, schema: type) -> object:
        if schema is NarrativeSummary:
            return NarrativeSummary(summary="Stub narrative summary.")
        return mock.return_value

    mock = MagicMock(return_value=FULLY_SPECIFIED_FIELDS, side_effect=_respond)
    monkeypatch.setattr(BedrockClient, "extract_structured", mock)
    return mock


@pytest.fixture
def client(extract_structured: MagicMock) -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _start_campaign(client: TestClient) -> str:
    response = client.post(
        "/campaigns", json={"user_request": "Promote my coffee shop", "organization_id": "org-1", "organization_max_budget": 1000.0}
    )
    assert response.status_code == 202
    return response.json()["workflow_id"]


class TestStartCampaign:
    def test_fully_specified_request_reaches_awaiting_review(self, client: TestClient) -> None:
        workflow_id = _start_campaign(client)

        status_response = client.get(f"/campaigns/{workflow_id}")

        assert status_response.json()["status"] == WorkflowStatus.AWAITING_REVIEW.value
        assert status_response.json()["campaign_proposal"]["brand"] == "Acme Coffee"
        assert status_response.json()["performance_insights"]["narrative_summary"] == "Stub narrative summary."

    def test_partial_request_pauses_for_intake_form(self, client: TestClient, extract_structured: MagicMock) -> None:
        extract_structured.return_value = ExtractedIntakeFields(brand="Acme Coffee")

        workflow_id = _start_campaign(client)
        status_response = client.get(f"/campaigns/{workflow_id}")

        assert status_response.json()["status"] == WorkflowStatus.AWAITING_INTAKE_FORM.value
        missing = status_response.json()["intake_form_request"]["missing_fields"]
        assert "goal" in missing
        assert "brand" not in missing

    def test_extraction_is_called_exactly_once_across_a_pause(
        self, client: TestClient, extract_structured: MagicMock
    ) -> None:
        # Regression check for the redundant-Bedrock-call trap: anything before
        # an interrupt() inside the *same* node replays on every resume, so
        # extraction lives in its own node (intake_extract) that commits
        # before intake_form (the one that pauses) ever runs.
        extract_structured.return_value = ExtractedIntakeFields(brand="Acme Coffee")
        workflow_id = _start_campaign(client)

        client.post(
            f"/campaigns/{workflow_id}/intake-form",
            json={
                "user_request": "ignored",
                "goal": "awareness",
                "budget": {"total_budget": 500.0},
                "campaign_timing": {"start_date": "2026-07-01"},
                "locations": [{"country": "USA"}],
                "target_audience": {"age_min": 18, "age_max": 35},
                "platforms": ["meta"],
            },
        )

        extraction_calls = [call for call in extract_structured.call_args_list if call.args[1] is ExtractedIntakeFields]
        assert len(extraction_calls) == 1


class TestIntakeForm:
    def test_resolves_in_multiple_rounds_via_separate_interrupts(
        self, client: TestClient, extract_structured: MagicMock
    ) -> None:
        extract_structured.return_value = ExtractedIntakeFields()
        workflow_id = _start_campaign(client)

        first_status = client.get(f"/campaigns/{workflow_id}").json()
        assert first_status["status"] == WorkflowStatus.AWAITING_INTAKE_FORM.value
        assert set(first_status["intake_form_request"]["missing_fields"]) == {
            "brand", "goal", "budget", "campaign_timing", "locations", "target_audience", "platforms"
        }

        client.post(
            f"/campaigns/{workflow_id}/intake-form", json={"user_request": "ignored", "brand": "Acme Coffee", "goal": "awareness"}
        )

        second_status = client.get(f"/campaigns/{workflow_id}").json()
        assert second_status["status"] == WorkflowStatus.AWAITING_INTAKE_FORM.value
        assert "brand" not in second_status["intake_form_request"]["missing_fields"]
        assert "budget" in second_status["intake_form_request"]["missing_fields"]

        client.post(
            f"/campaigns/{workflow_id}/intake-form",
            json={
                "user_request": "ignored",
                "budget": {"total_budget": 500.0},
                "campaign_timing": {"start_date": "2026-07-01"},
                "locations": [{"country": "USA"}],
                "target_audience": {"age_min": 18, "age_max": 35},
                "platforms": ["meta"],
            },
        )

        final_status = client.get(f"/campaigns/{workflow_id}").json()
        assert final_status["status"] == WorkflowStatus.AWAITING_REVIEW.value

    def test_wrong_status_returns_409(self, client: TestClient) -> None:
        workflow_id = _start_campaign(client)

        response = client.post(f"/campaigns/{workflow_id}/intake-form", json={"user_request": "ignored"})

        assert response.status_code == 409


class TestGetCampaignStatus:
    def test_unknown_workflow_returns_404(self, client: TestClient) -> None:
        response = client.get("/campaigns/does-not-exist")

        assert response.status_code == 404

    def test_returns_classified_intent_once_intake_has_run(self, client: TestClient) -> None:
        workflow_id = _start_campaign(client)

        status_response = client.get(f"/campaigns/{workflow_id}")

        assert status_response.json()["intent"] == Intent.CAMPAIGN_PLANNING.value


class TestPerformanceAnalysisIntent:
    def test_runs_performance_analyst_only_and_completes(
        self, client: TestClient, extract_structured: MagicMock
    ) -> None:
        extract_structured.return_value = ExtractedIntakeFields(intent=Intent.PERFORMANCE_ANALYSIS, brand="Acme Coffee")

        workflow_id = _start_campaign(client)
        status_response = client.get(f"/campaigns/{workflow_id}").json()

        assert status_response["intent"] == Intent.PERFORMANCE_ANALYSIS.value
        assert status_response["status"] == WorkflowStatus.COMPLETED.value
        assert status_response["performance_insights"] is not None
        assert status_response["campaign_proposal"] is None

    def test_pauses_for_brand_only_when_missing(self, client: TestClient, extract_structured: MagicMock) -> None:
        extract_structured.return_value = ExtractedIntakeFields(intent=Intent.PERFORMANCE_ANALYSIS)

        workflow_id = _start_campaign(client)
        status_response = client.get(f"/campaigns/{workflow_id}").json()

        assert status_response["status"] == WorkflowStatus.AWAITING_INTAKE_FORM.value
        assert status_response["intake_form_request"]["missing_fields"] == ["brand"]

        client.post(f"/campaigns/{workflow_id}/intake-form", json={"user_request": "ignored", "brand": "Acme Coffee"})

        final_status = client.get(f"/campaigns/{workflow_id}").json()
        assert final_status["status"] == WorkflowStatus.COMPLETED.value


class TestCreativeAssetIntent:
    def test_surfaces_not_implemented_stub_as_failed(
        self, extract_structured: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # stub_agent_implementations (autouse) patches CreativeDirectorAgent.run
        # to a canned success for the campaign_planning chain's tests, before
        # the graph is even built. Undo that here, *before* building our own
        # app/TestClient below, so the graph node captures the agent's real
        # run() - which still just raises NotImplementedError (see
        # agents/creative_director.py) - rather than the already-bound stub.
        def _unimplemented_run(self: CreativeDirectorAgent, state: dict) -> dict:
            raise NotImplementedError

        monkeypatch.setattr(CreativeDirectorAgent, "run", _unimplemented_run)
        extract_structured.return_value = ExtractedIntakeFields(intent=Intent.CREATIVE_ASSET, brand="Acme Coffee")

        with TestClient(create_app()) as client:
            workflow_id = _start_campaign(client)
            status_response = client.get(f"/campaigns/{workflow_id}").json()

        assert status_response["status"] == WorkflowStatus.FAILED.value
        assert status_response["intent"] == Intent.CREATIVE_ASSET.value


class TestApproveCampaign:
    def test_persists_and_returns_campaign_id(self, client: TestClient, stub_supabase_methods: MagicMock) -> None:
        workflow_id = _start_campaign(client)
        assert client.get(f"/campaigns/{workflow_id}").json()["status"] == WorkflowStatus.AWAITING_REVIEW.value

        response = client.post(f"/campaigns/{workflow_id}/approve")

        assert response.status_code == 200
        assert response.json()["status"] == WorkflowStatus.APPROVED.value
        assert response.json()["persisted_campaign_id"] == "campaign-123"
        stub_supabase_methods.assert_called_once()

    def test_wrong_status_returns_409(self, client: TestClient) -> None:
        workflow_id = _start_campaign(client)
        client.post(f"/campaigns/{workflow_id}/approve")

        response = client.post(f"/campaigns/{workflow_id}/approve")

        assert response.status_code == 409


class TestRejectCampaign:
    def test_marks_rejected_without_persisting(self, client: TestClient, stub_supabase_methods: MagicMock) -> None:
        workflow_id = _start_campaign(client)

        response = client.post(f"/campaigns/{workflow_id}/reject")

        assert response.status_code == 200
        assert response.json()["status"] == WorkflowStatus.REJECTED.value
        stub_supabase_methods.assert_not_called()

    def test_wrong_status_returns_409(self, client: TestClient) -> None:
        workflow_id = _start_campaign(client)
        client.post(f"/campaigns/{workflow_id}/reject")

        response = client.post(f"/campaigns/{workflow_id}/reject")

        assert response.status_code == 409


class TestFailureHandling:
    def test_agent_exception_surfaces_as_failed_status(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CampaignPlannerAgent, "plan", lambda self, campaign_brief, performance_insights: (_ for _ in ()).throw(RuntimeError("boom")))

        workflow_id = _start_campaign(client)
        status_response = client.get(f"/campaigns/{workflow_id}")

        assert status_response.json()["status"] == WorkflowStatus.FAILED.value
        assert "boom" in status_response.json()["error"]
