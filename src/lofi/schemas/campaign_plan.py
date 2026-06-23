"""The finalized CampaignPlan produced by the Campaign Planner Agent.

Kept in its own module (rather than campaign_planner.py) because it is a
shared dependency of both the Campaign Planner and QA Agent schemas, and
keeping it here avoids a circular import between the two.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from lofi.schemas.common import AudienceSpec, BudgetSpec, CampaignGoal, CampaignTiming, Location, Platform


class CampaignPlan(BaseModel):
    goal: CampaignGoal = Field(description="Finalized campaign goal")
    campaign_type: str = Field(description="Campaign type, e.g. 'always-on', 'seasonal', 'launch'")
    objective: str = Field(description="Specific platform-level objective, e.g. 'conversions', 'reach'")
    audience: AudienceSpec = Field(description="Finalized target audience for the campaign")
    platforms: List[Platform] = Field(description="Finalized list of platforms the campaign will run on")
    locations: List[Location] = Field(description="Finalized target locations for the campaign")
    budget: BudgetSpec = Field(description="Finalized budget, including daily cap and platform split")
    timing: CampaignTiming = Field(description="Finalized campaign flight dates")
