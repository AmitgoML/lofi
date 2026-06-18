import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Sequence

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelResponse


class SaveFileOutput(BaseModel):
    """Output model for individual file metadata."""

    file_name: str = Field(..., description="Name of the saved file")
    file_path: str = Field(..., description="Public URL or storage path of the file")
    file_type: str = Field(..., description="MIME type or file type")
    asset_id: Optional[str] = Field(
        None, description="Optional creative asset ID (public.creative_assets.asset_id)"
    )
    job_id: Optional[str] = Field(
        None, description="Optional job identifier associated with the file/task"
    )


class JSONOutput(BaseModel):
    """Output model for structured JSON payloads."""

    json_type: str = Field(..., description="Schema name for the JSON payload")
    json_data: Any = Field(..., description="JSON-serializable payload data")


class FileAgentOutput(BaseModel):
    """Standard output model for all agents, including optional file metadata."""

    message: str = Field(..., description="Status message or summary")
    files: List[SaveFileOutput] = Field(
        default_factory=list, description="List of uploaded file metadata"
    )
    jsons: List[JSONOutput] = Field(
        default_factory=list, description="List of structured JSON payloads"
    )


class SearchResult(BaseModel):
    """Result from semantic search containing Q&A information."""

    id: str = Field(description="Unique identifier for the knowledge item")
    questions: List[str] = Field(description="Related questions that match the search")
    answer: str = Field(description="The answer to the questions")
    implications: List[str] = Field(
        description="Actionable implications and recommendations"
    )
    source: str = Field(description="Source URL or reference for the information")
    similarity_score: float = Field(
        description="Semantic similarity score (0-1, higher is better)"
    )


class UserOrgProfile(BaseModel):
    """Joined profile + organization view for richer user context.

    Known fields are typed for IDE support. Extra fields from the DB are
    allowed through so new columns are automatically visible to agents.
    """

    model_config = ConfigDict(extra="allow")

    org_id: Optional[str] = Field(None, description="Organization ID")
    first_name: Optional[str] = Field(None, description="The user's first name")
    last_name: Optional[str] = Field(None, description="The user's last name")
    website_url: Optional[str] = Field(None, description="Website URL")
    company_name: Optional[str] = Field(None, description="Organization name")
    industry: Optional[str] = Field(None, description="Organization industry")
    states: Optional[str] = Field(
        None, description="State or region (e.g., US state code)"
    )
    # Brand fields (all nullable)
    brand_name: Optional[str] = Field(None, description="Brand name")
    description: Optional[str] = Field(None, description="Brand description")
    tone_of_voice: Optional[str] = Field(None, description="Preferred tone of voice")
    purpose: Optional[str] = Field(None, description="Brand purpose")
    mission_vision: Optional[str] = Field(None, description="Mission and vision")
    core_values: Optional[str] = Field(None, description="Core values")
    audience: Optional[str] = Field(None, description="Primary audience")
    positioning: Optional[str] = Field(None, description="Positioning statement")
    design_elements: Optional[str] = Field(None, description="Design elements guidance")
    messaging_pillars: Optional[str] = Field(None, description="Messaging pillars")
    copywriting_tone: Optional[str] = Field(None, description="Copywriting tone")


def get_brand_name(profiles: "Optional[Sequence[UserOrgProfile]]") -> Optional[str]:
    """Return the first profile's brand_name, falling back to company_name, or None."""
    if not profiles:
        return None
    return getattr(profiles[0], "brand_name", None) or getattr(profiles[0], "company_name", None)


class ModelFileResponse:
    """Custom ModelResponse subclass for file messages."""

    def __init__(self, parts, timestamp=None):
        # Import here to avoid circular imports
        from pydantic_ai.messages import ModelResponse

        # Create a ModelResponse instance
        self._model_response = ModelResponse(parts=parts, timestamp=timestamp)

    def __getattr__(self, name):
        # Delegate all attributes to the underlying ModelResponse
        return getattr(self._model_response, name)

    def __setattr__(self, name, value):
        # Handle our own attributes
        if name == "_model_response":
            super().__setattr__(name, value)
        else:
            # Delegate to the underlying ModelResponse
            setattr(self._model_response, name, value)


@dataclass
class ChatDeps:
    system: Optional[str] = None
    user_id: Optional[str] = None
    user_profiles: Optional[List["UserOrgProfile"]] = None
    message_history: Optional[List["ModelMessage"]] = None
    user_location: Optional[str] = None
    request_type: Optional[str] = None  # Type of request: "image", "video", or None
    request_params: Optional[Dict[str, Any]] = (
        None  # Parameters for image/video generation
    )
    attachments: Optional[List[Dict[str, Any]]] = (
        None  # Uploaded file attachments for image modification
    )

    # Shared tool state
    user_profiles_loaded: bool = False

    # Campaign Planner state
    campaign_planning_state: Optional[Any] = None
    planning_phase_complete: bool = False
    campaign_interview_used: bool = False
    campaign_interview_calls: int = 0
    draft_emitted: bool = False
    campaign_planner_pending_jsons: Optional[list] = None

    # Creative Director state
    creative_director_state: Optional[Any] = None
    creative_interview_used: bool = False
    creative_interview_ready: bool = False
    creative_interview_calls: int = 0
    creative_route: Optional[Dict[str, Any]] = None
    creative_execute_used: bool = False
    creative_router_calls: int = 0
    _logged_web_tools: bool = False
    user_location: Optional[str] = (
        None  # User's position/context on the website (e.g., /dashboard/campaigns)
    )
    
    # Frontend page context (parsed ChatContext dict)
    context: Optional[Dict[str, Any]] = None
    # Full brand data from get_full_brand (cached per brand_id)
    brand_context: Optional[Dict[str, Any]] = None
    # Query understanding output (intent, entities, retrieved context)
    query_context: Optional[Dict[str, Any]] = None

    # Tools push human-readable status strings here; the streaming generator
    # drains this queue and emits tool_status events to the frontend.
    status_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
