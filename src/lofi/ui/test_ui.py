"""Manual test UI for the intent router.

Talks to the running FastAPI app over HTTP only - it never imports the graph,
agents, or any client wiring directly, so this exercises the same surface a
real client would (and stays correct if the backend's internals change).

Run the API first:
    uv run uvicorn lofi.main:app --reload --app-dir src

Then, in another terminal:
    uv run --group ui streamlit run src/lofi/ui/test_ui.py
"""

import os
import time

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.environ.get("LOFI_API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="Lofi Intent Router - Test UI", layout="centered")


def start_workflow(base_url: str, user_request: str, organization_id: str, organization_max_budget: float) -> dict:
    response = requests.post(
        f"{base_url}/campaigns",
        json={
            "user_request": user_request,
            "organization_id": organization_id,
            "organization_max_budget": organization_max_budget,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_status(base_url: str, workflow_id: str) -> dict:
    response = requests.get(f"{base_url}/campaigns/{workflow_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def submit_intake_form(base_url: str, workflow_id: str, submission: dict) -> dict:
    response = requests.post(f"{base_url}/campaigns/{workflow_id}/intake-form", json=submission, timeout=10)
    response.raise_for_status()
    return response.json()


def approve(base_url: str, workflow_id: str) -> dict:
    response = requests.post(f"{base_url}/campaigns/{workflow_id}/approve", timeout=30)
    response.raise_for_status()
    return response.json()


def reject(base_url: str, workflow_id: str) -> dict:
    response = requests.post(f"{base_url}/campaigns/{workflow_id}/reject", timeout=10)
    response.raise_for_status()
    return response.json()


def poll_until_settled(base_url: str, workflow_id: str) -> dict:
    """Starting a workflow and submitting the intake form both return 202
    immediately (see api/routes.py) - the graph keeps running as a FastAPI
    background task, so polling is how a real client would learn what
    happened."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    with st.spinner("Running..."):
        while time.monotonic() < deadline:
            status = get_status(base_url, workflow_id)
            if status["status"] != "processing":
                return status
            time.sleep(POLL_INTERVAL_SECONDS)
    return get_status(base_url, workflow_id)


def render_intake_form(base_url: str, workflow_id: str, missing_fields: list[str]) -> None:
    st.subheader("Intake form")
    st.caption(f"Missing fields: {', '.join(missing_fields)}")
    with st.form("intake_form"):
        values: dict = {}
        if "brand" in missing_fields:
            values["brand"] = st.text_input("Brand")
        if "goal" in missing_fields:
            values["goal"] = st.selectbox(
                "Goal", ["awareness", "traffic", "conversions", "engagement", "lead_gen", "app_installs"]
            )
        if "budget" in missing_fields:
            total_budget = st.number_input("Total budget", min_value=0.0, value=1000.0)
            values["budget"] = {"total_budget": total_budget}
        if "campaign_timing" in missing_fields:
            start_date = st.date_input("Start date")
            values["campaign_timing"] = {"start_date": start_date.isoformat()}
        if "locations" in missing_fields:
            country = st.text_input("Target country", value="USA")
            values["locations"] = [{"country": country}]
        if "target_audience" in missing_fields:
            age_min, age_max = st.slider("Target age range", 13, 90, (18, 45))
            values["target_audience"] = {"age_min": age_min, "age_max": age_max}
        if "platforms" in missing_fields:
            values["platforms"] = st.multiselect("Platforms", ["meta", "google", "tiktok", "spotify"])

        submitted = st.form_submit_button("Submit")

    if submitted:
        submit_intake_form(base_url, workflow_id, {"user_request": "ignored", **values})
        st.session_state["last_status"] = poll_until_settled(base_url, workflow_id)
        st.rerun()


def render_creative_assets(assets: list[dict]) -> None:
    """Render any image creative assets inline, by local file path/URL.

    asset_url points at a local path while CreativeDirectorAgent is still
    backed by produce_static_sample() (see agents/creative_director.py) -
    st.image() accepts that directly as long as the UI runs on the same
    machine as the API.
    """
    for asset in assets:
        if asset.get("creative_format") == "image":
            st.image(asset["asset_url"], caption=f"{asset.get('platform', '')} creative")


def render_human_review(base_url: str, workflow_id: str, campaign_proposal: dict | None) -> None:
    st.subheader("Human review")
    if campaign_proposal:
        render_creative_assets(campaign_proposal.get("creative_assets", []))
    st.json(campaign_proposal)
    col1, col2 = st.columns(2)
    if col1.button("Approve"):
        st.session_state["approval_result"] = approve(base_url, workflow_id)
        st.session_state["last_status"] = get_status(base_url, workflow_id)
        st.rerun()
    if col2.button("Reject"):
        reject(base_url, workflow_id)
        st.session_state["last_status"] = get_status(base_url, workflow_id)
        st.rerun()


st.title("Lofi Intent Router - Test UI")
st.caption(
    "Calls the FastAPI app's /campaigns endpoints to exercise campaign_planning / "
    "performance_analysis / creative_asset routing - start the API with "
    "`uv run uvicorn lofi.main:app --reload` first."
)

api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)

with st.form("start_form"):
    user_request = st.text_area(
        "User request",
        placeholder="e.g. 'How did our Meta ads perform last month?' or 'Plan a summer campaign for Acme Coffee'",
    )
    organization_id = st.text_input("Organization ID", value="org-1")
    organization_max_budget = st.number_input("Organization max budget", min_value=0.0, value=5000.0)
    start_submitted = st.form_submit_button("Start workflow")

if start_submitted and user_request.strip():
    try:
        started = start_workflow(api_base_url, user_request, organization_id, organization_max_budget)
        st.session_state["workflow_id"] = started["workflow_id"]
        st.session_state.pop("approval_result", None)
        st.session_state["last_status"] = poll_until_settled(api_base_url, started["workflow_id"])
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {api_base_url}: {exc}")

workflow_id = st.session_state.get("workflow_id")
if workflow_id:
    st.divider()
    st.caption(f"workflow_id: {workflow_id}")

    status = st.session_state.get("last_status") or get_status(api_base_url, workflow_id)

    if status.get("intent"):
        st.metric("Classified intent", status["intent"])

    if status["status"] == "awaiting_intake_form":
        render_intake_form(api_base_url, workflow_id, status["intake_form_request"]["missing_fields"])
    elif status["status"] == "awaiting_review":
        render_human_review(api_base_url, workflow_id, status.get("campaign_proposal"))
    elif status["status"] == "failed":
        st.error(status.get("error") or "Workflow failed.")
    else:
        st.subheader(f"Status: {status['status']}")
        if status.get("performance_insights"):
            st.write("Performance insights")
            st.json(status["performance_insights"])
        if status.get("creative_director_output"):
            st.write("Creative director output")
            render_creative_assets(status["creative_director_output"].get("assets", []))
            st.json(status["creative_director_output"])
        if status.get("campaign_proposal"):
            st.write("Campaign proposal")
            render_creative_assets(status["campaign_proposal"].get("creative_assets", []))
            st.json(status["campaign_proposal"])
        if st.session_state.get("approval_result", {}).get("persisted_campaign_id"):
            st.success(f"Persisted campaign ID: {st.session_state['approval_result']['persisted_campaign_id']}")
