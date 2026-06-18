import os
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


class MockAgentClass:
    """Mock agent class that provides the required class methods."""

    # Stores the deps instance received by pre_run_check for assertion in tests.
    last_pre_run_deps = None

    @classmethod
    def reminder_header(cls):
        return "Mock Agent"

    @classmethod
    def create(cls):
        return AsyncMock()

    @classmethod
    async def pre_run_check(cls, deps):
        MockAgentClass.last_pre_run_deps = deps
        return None

    @classmethod
    async def get_streaming_agent(cls, deps):
        return None


class TestChatStreamRequestType:
    """Test cases for chat stream endpoint with request_type parameter."""

    def test_chat_stream_with_image_request_type(self, client: TestClient):
        """Test that request_type='image' routes to image agent."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return image agent
            mock_image_agent = AsyncMock()
            mock_image_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_image_agent)

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create a sunset landscape",
                    "request_type": "image",
                    "request_params": {"size": "landscape", "quality": "high"},
                },
            )

            assert response.status_code == 200

            # Verify that route_request was called with request_type
            mock_route.assert_called_once()
            call_args = mock_route.call_args
            assert call_args[1]["request_type"] == "image"

    def test_chat_stream_with_none_request_type(self, client: TestClient):
        """Test that request_type=None uses normal routing."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return support agent
            mock_support_agent = AsyncMock()
            mock_support_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("support", MockAgentClass, mock_support_agent)

            response = client.post(
                "/chat/stream",
                json={"message": "Help me with billing", "request_type": None},
            )

            assert response.status_code == 200

            # Verify that route_request was called with request_type=None
            mock_route.assert_called_once()
            call_args = mock_route.call_args
            assert call_args[1]["request_type"] is None

    def test_chat_stream_without_request_type(self, client: TestClient):
        """Test that missing request_type uses normal routing."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return lucy agent
            mock_lucy_agent = AsyncMock()
            mock_lucy_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("lucy", MockAgentClass, mock_lucy_agent)

            response = client.post(
                "/chat/stream", json={"message": "Give me marketing advice"}
            )

            assert response.status_code == 200

            # Verify that route_request was called with request_type=None (default)
            mock_route.assert_called_once()
            call_args = mock_route.call_args
            assert call_args[1]["request_type"] is None

    def test_chat_stream_with_request_params(self, client: TestClient):
        """Test that request_params are passed to ChatDeps."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return image agent
            mock_image_agent = AsyncMock()
            mock_image_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_image_agent)

            request_params = {
                "size": "portrait",
                "quality": "high",
                "background": "opaque",
            }

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create a portrait",
                    "request_type": "image",
                    "request_params": request_params,
                },
            )

            assert response.status_code == 200

            # Verify that the agent was called with ChatDeps containing request_params
            mock_image_agent.run_stream.assert_called_once()
            call_args = mock_image_agent.run_stream.call_args
            deps = call_args[1]["deps"]
            assert deps.request_params == request_params
            assert deps.request_type == "image"

    def test_chat_stream_with_user_location(self, client: TestClient):
        """Test that user_location is passed to ChatDeps."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return support agent
            mock_support_agent = AsyncMock()
            mock_support_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("support", MockAgentClass, mock_support_agent)

            user_location = "/dashboard/campaigns"

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Help me with this page",
                    "user_location": user_location,
                },
            )

            assert response.status_code == 200

            # Verify that the agent was called with ChatDeps containing user_location
            mock_support_agent.run_stream.assert_called_once()
            call_args = mock_support_agent.run_stream.call_args
            deps = call_args[1]["deps"]
            assert deps.user_location == user_location

    def test_chat_stream_with_all_parameters(self, client: TestClient):
        """Test that all new parameters are passed correctly."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return image agent
            mock_image_agent = AsyncMock()
            mock_image_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_image_agent)

            request_params = {"size": "landscape", "quality": "medium"}
            user_location = "/settings/billing"

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create a landscape image",
                    "request_type": "image",
                    "request_params": request_params,
                    "user_location": user_location,
                    "session_id": "test-session",
                },
            )

            assert response.status_code == 200

            # Verify that the agent was called with all parameters
            mock_image_agent.run_stream.assert_called_once()
            call_args = mock_image_agent.run_stream.call_args
            deps = call_args[1]["deps"]
            assert deps.request_type == "image"
            assert deps.request_params == request_params
            assert deps.user_location == user_location

    def test_chat_stream_response_format(self, client: TestClient):
        """Test that the response format includes expected fields."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return image agent
            mock_image_agent = AsyncMock()
            mock_image_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_image_agent)

            response = client.post(
                "/chat/stream",
                json={"message": "Create an image", "request_type": "image"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/plain; charset=utf-8"

            # Check that response contains expected status messages
            response_text = response.text
            assert '"type": "status"' in response_text
            assert '"type": "delta"' in response_text

    def test_chat_stream_invalid_request_type(self, client: TestClient):
        """Test that invalid request_type values are rejected by validation."""
        # This test expects a 422 error due to Pydantic validation
        response = client.post(
            "/chat/stream",
            json={"message": "Test message", "request_type": "invalid_type"},
        )

        # The request should fail validation because "invalid_type" is not in the allowed values
        assert response.status_code == 422

    def test_chat_stream_empty_request_params(self, client: TestClient):
        """Test that empty request_params are handled correctly."""
        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            # Mock the routing to return image agent
            mock_image_agent = AsyncMock()
            mock_image_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_image_agent)

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create an image",
                    "request_type": "image",
                    "request_params": {},
                },
            )

            assert response.status_code == 200

            # Verify that empty request_params are passed correctly
            mock_image_agent.run_stream.assert_called_once()
            call_args = mock_image_agent.run_stream.call_args
            deps = call_args[1]["deps"]
            assert deps.request_params == {}

    def test_chat_stream_single_deps_instance(self, client: TestClient):
        """Test that pre_run_check and run_stream receive the same ChatDeps instance."""
        MockAgentClass.last_pre_run_deps = None

        with patch("lucy.agents.router_agent.RouterAgent.route_request") as mock_route:
            mock_agent = AsyncMock()
            mock_agent.run_stream.return_value.__aenter__.return_value.stream_output.return_value = (
                []
            )

            mock_route.return_value = ("image", MockAgentClass, mock_agent)

            request_params = {"size": "landscape", "quality": "high"}

            response = client.post(
                "/chat/stream",
                json={
                    "message": "Create a landscape",
                    "request_type": "image",
                    "request_params": request_params,
                },
            )

            assert response.status_code == 200

            # pre_run_check should have received a fully-populated ChatDeps
            assert MockAgentClass.last_pre_run_deps is not None
            assert MockAgentClass.last_pre_run_deps.request_params == request_params
            assert MockAgentClass.last_pre_run_deps.request_type == "image"

            # run_stream should receive the exact same deps instance
            mock_agent.run_stream.assert_called_once()
            run_stream_deps = mock_agent.run_stream.call_args[1]["deps"]
            assert run_stream_deps is MockAgentClass.last_pre_run_deps

    def test_chat_stream_malformed_request_params(self, client: TestClient):
        """Test that malformed request_params are handled gracefully."""
        # This test expects a 422 error due to Pydantic validation
        response = client.post(
            "/chat/stream",
            json={
                "message": "Create an image",
                "request_type": "image",
                "request_params": "not_a_dict",  # This should be handled gracefully
            },
        )

        # The request should fail validation
        assert response.status_code == 422
