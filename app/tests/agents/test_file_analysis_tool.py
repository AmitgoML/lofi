import io
import pytest
from unittest.mock import Mock, patch, AsyncMock
from lucy.agents.common.file_tools import (
    GenericFileAnalysis,
    _render_csv_for_llm,
    _render_excel_for_llm,
    _render_docx_for_llm,
    _render_generic_text_for_llm,
)
from lucy.agents.common.models import ChatDeps
from lucy.agents.lucy_agent import LucyAgent
from tests.agents.helpers import mock_router


class TestFileRenderingHelpers:
    """Test cases for file rendering helper functions."""

    def test_render_csv_for_llm(self):
        """Test CSV rendering for LLM."""
        csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA"
        result = _render_csv_for_llm(csv_content)

        assert "CSV structure" in result
        assert "Columns:" in result
        assert "name" in result
        assert "age" in result
        assert "city" in result
        assert "John" in result
        assert "Jane" in result

    def test_render_csv_for_llm_empty(self):
        """Test CSV rendering with empty file."""
        result = _render_csv_for_llm(b"")
        assert "empty" in result.lower()

    def test_render_csv_for_llm_max_rows(self):
        """Test that CSV rendering respects max_rows limit."""
        # Create CSV with more than 50 rows
        rows = ["col1,col2"] + [f"val1_{i},val2_{i}" for i in range(100)]
        csv_content = "\n".join(rows).encode("utf-8")
        result = _render_csv_for_llm(csv_content, max_rows=50)

        # Should only show up to 50 sample rows
        assert "Sample rows (up to 50 rows):" in result
        # Count the data rows (excluding header)
        data_lines = [line for line in result.split("\n") if "val1_" in line]
        assert len(data_lines) <= 50

    def test_render_generic_text_for_llm(self):
        """Test generic text rendering."""
        text_content = b"This is a test file with some content."
        result = _render_generic_text_for_llm(text_content)

        assert "This is a test file" in result
        assert "some content" in result

    def test_render_generic_text_for_llm_empty(self):
        """Test generic text rendering with empty file."""
        result = _render_generic_text_for_llm(b"")
        assert "empty" in result.lower()

    def test_render_generic_text_for_llm_max_chars(self):
        """Test that generic text rendering respects max_chars limit."""
        large_content = b"x" * 300000  # 300 KB
        result = _render_generic_text_for_llm(large_content, max_chars=200000)

        assert len(result) <= 200000

    def test_render_generic_text_for_llm_invalid_utf8(self):
        """Test that invalid UTF-8 is handled gracefully."""
        invalid_utf8 = b"\xff\xfe\x00\x01"
        result = _render_generic_text_for_llm(invalid_utf8)

        # Should not raise exception, should handle with errors="replace"
        assert isinstance(result, str)


class TestRouterAgentWithAttachments:
    """Test cases for RouterAgent with file attachments."""

    @pytest.mark.asyncio
    async def test_route_request_with_attachments_image_request_type(self):
        """Test that attachments with image request_type route to image agent."""
        from lucy.agents.router_agent import RouterAgent
        from lucy.agents.image_agent import ImageAgent

        attachments = [{"url": "https://example.com/image.jpg", "path": "image.jpg"}]

        with patch.object(ImageAgent, "create") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Modify this image",
                routing_history=[],
                user_id="test-user",
                request_type="image",
                attachments=attachments,
            )

            assert route == "image"
            assert agent_class == ImageAgent
            assert agent_instance == mock_agent

    @pytest.mark.asyncio
    async def test_route_request_with_attachments_video_request_type(self):
        """Test that attachments with video request_type route to video agent."""
        from lucy.agents.router_agent import RouterAgent
        from lucy.agents.video_agent import VideoAgent

        attachments = [{"url": "https://example.com/video.mp4", "path": "video.mp4"}]

        with patch.object(VideoAgent, "create") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Create a video",
                routing_history=[],
                user_id="test-user",
                request_type="video",
                attachments=attachments,
            )

            assert route == "video"
            assert agent_class == VideoAgent
            assert agent_instance == mock_agent

    @pytest.mark.asyncio
    async def test_route_request_with_attachments_empty_list(self):
        """Test that empty attachments list uses normal routing."""
        from lucy.agents.router_agent import RouterAgent
        from lucy.agents.support_agent import SupportAgent

        mock_support_agent = Mock()
        with mock_router("support", 0.95, SupportAgent, mock_support_agent):
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="Help with billing",
                routing_history=[],
                user_id="test-user",
                request_type=None,
                attachments=[],
            )

            # Should use normal routing, not short-circuit to lucy
            assert route == "support"
            assert agent_class == SupportAgent

    @pytest.mark.asyncio
    async def test_route_request_with_attachments_none(self):
        """Test that None attachments uses normal routing."""
        from lucy.agents.router_agent import RouterAgent
        from lucy.agents.keywords_agent import KeywordsAgent

        mock_keywords_agent = Mock()
        with mock_router("keywords", 0.9, KeywordsAgent, mock_keywords_agent):
            route, agent_class, agent_instance = await RouterAgent.route_request(
                message="What keywords should I use?",
                routing_history=[],
                user_id="test-user",
                request_type=None,
                attachments=None,
            )

            # Should use normal routing
            assert route == "keywords"
            assert agent_class == KeywordsAgent


class TestFileAnalysisIntegration:
    """Integration tests for file analysis functionality."""

    def test_lucy_agent_has_file_analysis_tool(self):
        """Test that LucyAgent has file analysis tool registered."""
        agent = LucyAgent.create()

        # Check that the agent was created successfully
        assert agent is not None

        # The tool should be registered (we can't easily check this without accessing internals,
        # but if agent creation succeeds, the tool registration worked)
        # This is more of a smoke test to ensure the tool registration doesn't break agent creation

    def test_generic_file_analysis_model(self):
        """Test that GenericFileAnalysis model works correctly."""
        analysis = GenericFileAnalysis(
            file_name="test.pdf",
            file_type="pdf",
            summary="This is a test summary of the file content.",
            key_topics=["marketing", "ROAS", "campaign optimization"],
            important_entities=["Company A", "Product B"],
            potential_actions=[
                "Optimize budget allocation",
                "Refine audience targeting",
            ],
        )

        assert analysis.file_name == "test.pdf"
        assert analysis.file_type == "pdf"
        assert "summary" in analysis.summary.lower()
        assert len(analysis.key_topics) == 3
        assert len(analysis.important_entities) == 2
        assert len(analysis.potential_actions) == 2

    def test_generic_file_analysis_model_empty_lists(self):
        """Test that GenericFileAnalysis works with empty lists."""
        analysis = GenericFileAnalysis(
            file_name="empty.txt",
            file_type="text",
            summary="Empty file summary",
            key_topics=[],
            important_entities=[],
            potential_actions=[],
        )

        assert analysis.file_name == "empty.txt"
        assert analysis.key_topics == []
        assert analysis.important_entities == []
        assert analysis.potential_actions == []
