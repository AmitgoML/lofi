import pytest
from unittest.mock import patch
from lucy.agents.video_agent import VideoAgent
from tests.agents.helpers import make_mock_ctx


class TestVideoAgentParameterProcessing:
    """Test cases for VideoAgent parameter processing functionality."""

    def test_process_video_params_no_request_params(self):
        """Test parameter processing when no request_params are provided."""
        seconds, size = VideoAgent._process_video_params(make_mock_ctx(None), 4, "1280x720")

        assert seconds == 4
        assert size == "1280x720"

    def test_process_video_params_with_request_params(self):
        """Test parameter processing with request_params provided."""
        seconds, size = VideoAgent._process_video_params(
            make_mock_ctx({"duration": 8, "ratio": "landscape"}), 4, "1280x720"
        )

        assert seconds == 8
        assert size == "1280x720"  # landscape maps to 1280x720

    def test_process_video_params_size_mapping(self):
        """Test size parameter mapping from descriptive words to pixel dimensions."""
        _, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": "auto"}))
        assert size == "1280x720"

        _, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": "landscape"}))
        assert size == "1280x720"

        _, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": "portrait"}))
        assert size == "720x1280"

    def test_process_video_params_size_validation(self):
        """Test that invalid sizes are corrected to valid values."""
        _, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": "123x456"}))
        assert size == "1280x720"

    def test_process_video_params_seconds_validation(self):
        """Test that seconds values are clamped to valid values (4, 8, 12)."""
        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 5}))
        assert seconds == 4

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 2}))
        assert seconds == 4

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 10}))
        assert seconds == 8

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 15}))
        assert seconds == 12

    def test_process_video_params_valid_seconds(self):
        """Test that valid seconds values (4, 8, 12) are preserved."""
        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 4}))
        assert seconds == 4

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 8}))
        assert seconds == 8

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 12}))
        assert seconds == 12

    def test_process_video_params_seconds_string_conversion(self):
        """Test that string seconds values are converted to integers."""
        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": "8"}))
        assert seconds == 8
        assert isinstance(seconds, int)

        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": 8.0}))
        assert seconds == 8
        assert isinstance(seconds, int)

    def test_process_video_params_seconds_invalid_string(self):
        """Test that invalid string seconds values default to default value."""
        seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": "invalid"}), 4, "1280x720")
        assert seconds == 4

    def test_process_video_params_partial_request_params(self):
        """Test parameter processing with only some request_params provided."""
        seconds, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": "portrait"}), 4, "1280x720")

        assert seconds == 4  # default
        assert size == "720x1280"  # portrait mapping

    def test_process_video_params_empty_request_params(self):
        """Test parameter processing with empty request_params."""
        seconds, size = VideoAgent._process_video_params(make_mock_ctx({}), 4, "1280x720")

        assert seconds == 4
        assert size == "1280x720"

    def test_process_video_params_custom_defaults(self):
        """Test parameter processing with custom default values."""
        seconds, size = VideoAgent._process_video_params(make_mock_ctx(None), 8, "720x1280")

        assert seconds == 8
        assert size == "720x1280"

    def test_process_video_params_logging(self):
        """Test that parameter processing logs correctly."""
        mock_ctx = make_mock_ctx({"duration": 8, "ratio": "portrait"})

        with patch("lucy.agents.video_agent.logger") as mock_logger:
            VideoAgent._process_video_params(mock_ctx)

            # Check that logging was called
            mock_logger.info.assert_called_once()
            log_call = mock_logger.info.call_args[0][0]
            assert "Using request_params" in log_call
            assert "seconds=8" in log_call
            assert "size=720x1280" in log_call

    def test_process_video_params_no_logging_when_no_request_params(self):
        """Test that no logging occurs when no request_params are provided."""
        mock_ctx = make_mock_ctx(None)

        with patch("lucy.agents.video_agent.logger") as mock_logger:
            VideoAgent._process_video_params(mock_ctx)

            # Check that no logging occurred
            mock_logger.info.assert_not_called()


class TestVideoAgentDefaults:
    """Test cases for VideoAgent default values."""

    def test_default_video_seconds(self):
        """Test that default video seconds is 4."""
        from lucy.agents.video_agent import DEFAULT_VIDEO_SECONDS

        assert DEFAULT_VIDEO_SECONDS == 4

    def test_default_video_size(self):
        """Test that default video size is 1280x720."""
        from lucy.agents.video_agent import DEFAULT_VIDEO_SIZE

        assert DEFAULT_VIDEO_SIZE == "1280x720"


class TestVideoAgentIntegration:
    """Integration tests for VideoAgent with parameter processing."""

    @pytest.mark.asyncio
    async def test_video_generation_tool_uses_request_params(self):
        """Test that video generation tool uses request_params when available."""
        seconds, size = VideoAgent._process_video_params(
            make_mock_ctx({"duration": 8, "ratio": "portrait"}), 4, "1280x720"
        )

        assert seconds == 8
        assert size == "720x1280"  # portrait mapping

    def test_video_size_valid_values(self):
        """Test that only valid size values are accepted."""
        valid_sizes = ["720x1280", "1280x720"]

        for valid_size in valid_sizes:
            _, size = VideoAgent._process_video_params(make_mock_ctx({"ratio": valid_size}))
            assert size in valid_sizes

    def test_video_seconds_valid_values(self):
        """Test that only valid seconds values are accepted (4, 8, 12)."""
        for valid_second in [4, 8, 12]:
            seconds, _ = VideoAgent._process_video_params(make_mock_ctx({"duration": valid_second}))
            assert seconds == valid_second
