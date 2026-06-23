"""Supabase persistence: campaigns and reference metric tables."""


class SupabaseClient:
    """Reads/writes campaigns, campaign_*_metrics, and workflow state tables."""

    def get_platform_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def get_location_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def get_audience_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def get_creative_metrics(self, organization_id: str) -> list[dict]:
        raise NotImplementedError

    def save_campaign(self, campaign_proposal: dict) -> str:
        raise NotImplementedError

    def save_workflow_state(self, workflow_id: str, state: dict) -> None:
        raise NotImplementedError

    def load_workflow_state(self, workflow_id: str) -> dict:
        raise NotImplementedError
