import asyncio
import os
from typing import List, Dict, Any


from loguru import logger
from pydantic_ai import Agent, RunContext
from pydantic import Field

from lucy.agents.common.models import ChatDeps, UserOrgProfile, SearchResult
from lucy.database import get_user_org_profiles
from lucy.kb import get_knowledge_base


def register_user_org_profiles_tool(agent: Agent) -> None:
    """Register tools used by the single knowledge agent."""

    @agent.tool
    async def get_user_org_profiles_tool(
        ctx: RunContext[ChatDeps],
    ) -> list[UserOrgProfile]:
        ctx.deps.status_queue.put_nowait("Loading your profile")
        user_id = ctx.deps.user_id
        if ctx.deps.user_profiles:
            ctx.deps.user_profiles_loaded = True
            return ctx.deps.user_profiles
        logger.info(f"Getting user org profiles for user id: {user_id}")
        if not user_id:
            ctx.deps.user_profiles = []
            ctx.deps.user_profiles_loaded = True
            return []
        try:
            timeout_s = float(os.getenv("PROFILE_TOOL_TIMEOUT_SEC", "2.0"))
            rows = await asyncio.wait_for(
                asyncio.to_thread(get_user_org_profiles, user_id), timeout=timeout_s
            )
            profiles = [UserOrgProfile(**row) for row in rows]
            ctx.deps.user_profiles = profiles
            ctx.deps.user_profiles_loaded = True
            return profiles
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout retrieving user org profiles for {user_id}; proceeding without personalization"
            )
            ctx.deps.user_profiles = []
            ctx.deps.user_profiles_loaded = True
            return []
        except Exception as e:
            logger.error(f"Error retrieving user org profiles for {user_id}: {str(e)}")
            ctx.deps.user_profiles = []
            ctx.deps.user_profiles_loaded = True
            return []


def register_semantic_search_tool(agent: Agent) -> None:
    @agent.tool
    async def semantic_search_tool(
        ctx: RunContext[ChatDeps],
        query: str = Field(description="The search query to find relevant knowledge"),
        top_k: int = Field(
            default=3, description="Number of top results to return (1-10)"
        ),
    ) -> List[SearchResult]:
        """
        Search the knowledge base using semantic similarity to find relevant Q&A information.

        This tool performs semantic search across marketing and business knowledge to provide
        answers and actionable implications based on your query.
        """
        ctx.deps.status_queue.put_nowait("Searching knowledge base")
        logger.info(f"Performing semantic search for query: '{query}' (top_k={top_k})")

        # Validate top_k parameter
        if top_k < 1:
            top_k = 1
        elif top_k > 10:
            top_k = 10

        try:
            # Get timeout from environment (default 5 seconds for semantic search)
            timeout_s = float(os.getenv("SEMANTIC_SEARCH_TIMEOUT_SEC", "5.0"))

            # Get the knowledge base instance and perform search with timeout
            kb = get_knowledge_base()
            search_results = await asyncio.wait_for(
                asyncio.to_thread(kb.search, query, top_k), timeout=timeout_s
            )

            # Convert results to SearchResult objects
            results = []
            for item, similarity_score in search_results:
                result = SearchResult(
                    id=item.id,
                    questions=item.questions,
                    answer=item.answer,
                    implications=item.implications,
                    source=item.source,
                    similarity_score=round(float(similarity_score), 3),
                )
                results.append(result)

            logger.info(f"Found {len(results)} relevant knowledge items")
            return results

        except asyncio.TimeoutError:
            logger.warning(
                f"Semantic search timeout for query '{query}'; returning empty results"
            )
            return []
        except Exception as e:
            logger.error(f"Error performing semantic search for '{query}': {str(e)}")
            return []
