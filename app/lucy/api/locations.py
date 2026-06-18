"""Location filter API — natural language search over a brand's location list.

POST /locations/filter
  - Stateless: no session context, nothing written to conversation history.
  - Auth: standard JWT via verify_jwt / extract_user_id.
  - The backend fetches brand_locations from the brands table; the caller
    only needs to supply the query string and a brand_id.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from lucy.agents.common.model_config import Models
from lucy.database.supabase_client import get_full_brand, get_user_org_profiles
from lucy.utils.auth import extract_user_id, verify_jwt

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LocationFilterRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language location query")
    brand_id: str = Field(..., description="UUID of the brand whose locations to search")


class LocationFilterResponse(BaseModel):
    matched_ids: List[str] = Field(
        description="brand_loc_-prefixed place_ids that match the query"
    )
    is_confident: bool = Field(
        description="True when the backend is confident the query mapped cleanly"
    )
    interpretation: str = Field(
        description="Human-readable summary of how the query was understood"
    )


# ---------------------------------------------------------------------------
# LLM extraction model (internal)
# ---------------------------------------------------------------------------


class _LocationMatch(BaseModel):
    matched_ids: List[str]
    is_confident: bool
    interpretation: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a location filter for a marketing platform. Given a natural language query and a JSON list of brand locations, return the subset of location IDs that match the query.

Each location in the list has:
- id: the place_id string (always starts with "brand_loc_")
- name: display name of the location
- place_name: full address string
- tags: optional list of tag strings (e.g. "New Store", "Florida", city/state tags)

Your task:
1. Identify which locations match the query based on name, address, and tags.
2. Return a JSON object with:
   - matched_ids: array of matching "id" values (use the exact id strings provided)
   - is_confident: true if you clearly understood the query and the matches are unambiguous, false if the query is vague or there's meaningful uncertainty
   - interpretation: a short human-readable string explaining how you understood the query (e.g. "New Stores tag + (Florida or Oregon state)")

Rules:
- Return ONLY locations from the provided list. Never invent IDs.
- matched_ids may be empty if nothing matches.
- Keep interpretation concise (one line, no emojis).
- Return only valid JSON matching the schema. No markdown, no extra keys.
""".strip()


# ---------------------------------------------------------------------------
# LLM agent (module-level singleton, created on first request)
# ---------------------------------------------------------------------------

_location_agent: Optional[Agent] = None


def _get_location_agent() -> Agent:
    global _location_agent
    if _location_agent is None:
        _location_agent = Agent(
            model=Models.AGENT_FAST,
            system_prompt=_SYSTEM_PROMPT,
            output_type=_LocationMatch,
            model_settings=ModelSettings(temperature=0.0, max_tokens=512),
        )
    return _location_agent


# ---------------------------------------------------------------------------
# Helper: flatten brand_locations JSONB into a simple list for the LLM
# ---------------------------------------------------------------------------


def _flatten_locations(brand_locations: Any) -> List[Dict[str, Any]]:
    """Convert the brands.brand_locations JSONB into a compact list for the LLM."""
    if not brand_locations:
        return []

    if isinstance(brand_locations, str):
        try:
            brand_locations = json.loads(brand_locations)
        except Exception:
            return []

    if not isinstance(brand_locations, list):
        return []

    out: List[Dict[str, Any]] = []
    for loc in brand_locations:
        if not isinstance(loc, dict):
            continue
        address = loc.get("address") or {}
        place_id = address.get("place_id") if isinstance(address, dict) else None
        if not place_id:
            continue
        out.append(
            {
                "id": place_id,
                "name": loc.get("name") or "",
                "place_name": address.get("place_name") or "",
                "tags": loc.get("tags") or [],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/filter",
    response_model=LocationFilterResponse,
    summary="Filter brand locations by natural language query",
)
async def filter_locations(
    req: LocationFilterRequest,
    payload: Dict[str, Any] = Depends(verify_jwt),
) -> LocationFilterResponse:
    """Return the brand locations that match a natural language query.

    Stateless — nothing is written to conversation history.
    The caller must own the brand (verified via org membership).
    """
    user_id = extract_user_id(payload)

    # --- Fetch brand (cached) --------------------------------------------------
    brand = get_full_brand(req.brand_id)
    if not brand:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brand not found",
        )

    # --- Authorisation: confirm the user's org owns this brand ----------------
    profiles = get_user_org_profiles(user_id)
    user_org_ids = {p.get("org_id") for p in (profiles or []) if p.get("org_id")}
    brand_org_id = brand.get("associated_organization_id")

    if not brand_org_id or brand_org_id not in user_org_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this brand",
        )

    # --- Build location list ---------------------------------------------------
    locations = _flatten_locations(brand.get("brand_locations"))
    if not locations:
        return LocationFilterResponse(
            matched_ids=[],
            is_confident=True,
            interpretation="Brand has no locations configured",
        )

    # --- Call LLM -------------------------------------------------------------
    user_message = json.dumps(
        {"query": req.query, "locations": locations}, ensure_ascii=False
    )

    try:
        result = await _get_location_agent().run(user_message)
        match: _LocationMatch = result.output
    except Exception as exc:
        logger.opt(exception=True).error(
            f"Location filter LLM call failed (brand={req.brand_id}): {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process location query",
        )

    # Validate: only return IDs that actually exist in the brand's list
    valid_ids = {loc["id"] for loc in locations}
    sanitised_ids = [mid for mid in match.matched_ids if mid in valid_ids]

    return LocationFilterResponse(
        matched_ids=sanitised_ids,
        is_confident=match.is_confident,
        interpretation=match.interpretation,
    )
