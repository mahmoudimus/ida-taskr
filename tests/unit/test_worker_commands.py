"""
Tests for WorkerBase command handling functionality.
"""

import asyncio  # Add asyncio for Event
import os
import sys
import typing  # Import typing for cast
from unittest.mock import Mock, patch

import pytest

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ida_taskr.utils import AsyncEventEmitter  # Import for spec
from ida_taskr.worker import ConnectionContext, WorkerBase


# Define a dummy emitter for testing purposes
class DummyAsyncEmitter(AsyncEventEmitter):
    def __init__(self, **kwargs):  # Accept arbitrary kwargs
        AsyncEventEmitter.__post_init__(self)  # Call base explicitly
        self.run_called = False
        self.shutdown_called = False
        self.custom_args = kwargs

    async def run(self):
        self.run_called = True
        # Simulate some work or emitting an event if needed for tests
        await asyncio.sleep(0)  # Yield control briefly
        return "dummy_run_result"

    async def shutdown(self):
        self.shutdown_called = True
        await asyncio.sleep(0)


class TestWorkerCommands:
    """Test suite for WorkerBase command handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.worker = WorkerBase(
            async_emitter_class=DummyAsyncEmitter, emitter_args={"test_arg": 123}
        )
        self.worker.setup()

        self.mock_conn = Mock(spec=ConnectionContext)

        self.mock_controller = Mock()
        # Link the mock controller to the actual emitter instance from the worker
        # This is important for side effects that operate on the emitter's events
        self.mock_controller.emitter = self.worker.emitter_instance

        # Configure side effects for mock_controller.pause and mock_controller.resume
        def _mock_controller_pause_side_effect(*args, **kwargs):
            if self.mock_controller.emitter:  # Should be self.worker.emitter_instance
                self.mock_controller.emitter.pause_evt.set()

        self.mock_controller.pause.side_effect = _mock_controller_pause_side_effect

        def _mock_controller_resume_side_effect(*args, **kwargs):
            if self.mock_controller.emitter:
                self.mock_controller.emitter.pause_evt.clear()

        self.mock_controller.resume.side_effect = _mock_controller_resume_side_effect

        # No specific side effect needed for stop for current assertions, just tracking calls.

    def test_worker_initialization(self):
        """Test WorkerBase initialization includes command handlers."""
        worker = WorkerBase(async_emitter_class=DummyAsyncEmitter, emitter_args={})
        worker.setup()

        assert "stop" in worker._commands
        assert "pause" in worker._commands
        assert "resume" in worker._commands
        assert not worker._running
        assert worker.emitter_instance is not None
        assert isinstance(worker.emitter_instance, DummyAsyncEmitter)  # Verify type
        emitter = typing.cast(DummyAsyncEmitter, worker.emitter_instance)
        assert not emitter.pause_evt.is_set()

    def test_handle_command_with_valid_command(self):
        """Test handling valid commands."""
        # Test stop command
        cmd = {"command": "stop"}
        # Simulate worker has been started and controller is set
        self.worker.controller = self.mock_controller

        result = self.worker.handle_command(cmd, self.mock_conn)

        assert not result  # stop should return False to exit
        self.mock_conn.send_message.assert_called_with(
            "status", "stopped", status="stopped"
        )
        self.mock_controller.stop.assert_called_once()  # Now this should be called

    def test_handle_command_with_invalid_command(self):
        """Test handling invalid/unknown commands."""
        cmd = {"command": "unknown_command"}
        result = self.worker.handle_command(cmd, self.mock_conn)

        assert result  # unknown commands should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_command_with_missing_command(self):
        """Test handling command dict without 'command' key."""
        cmd = {"other_field": "value"}
        result = self.worker.handle_command(cmd, self.mock_conn)

        assert result  # missing command should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_command_with_none_command(self):
        """Test handling command dict with None command."""
        cmd = {"command": None}
        result = self.worker.handle_command(cmd, self.mock_conn)

        assert result  # None command should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_stop_command(self):
        """Test stop command handling."""
        cmd = {"command": "stop"}
        # Ensure controller is mocked if _handle_stop interacts with it
        self.worker.controller = self.mock_controller

        result = self.worker._handle_stop(cmd, self.mock_conn)

        assert not result  # stop should return False
        assert not self.worker._running  # _running is set by this handler
        self.mock_conn.send_message.assert_called_with(
            "status", "stopped", status="stopped"
        )  # Ensure status kwarg
        if self.worker.controller:
            self.worker.controller.stop.assert_called_once()

    def test_handle_pause_command(self):
        """Test pause command handling."""
        cmd = {"command": "pause"}
        # Simulate worker being started, so controller exists
        self.worker.controller = self.mock_controller

        result = self.worker._handle_pause(cmd, self.mock_conn)

        assert result  # pause should return True to continue
        emitter = typing.cast(DummyAsyncEmitter, self.worker.emitter_instance)
        assert emitter.pause_evt.is_set()  # Check via emitter
        self.mock_conn.send_message.assert_called_with(
            "status", "paused", status="paused"
        )  # Ensure status kwarg
        self.mock_controller.pause.assert_called_once()

    def test_handle_resume_command(self):
        """Test resume command handling."""
        # First pause the worker by setting the event on the mock emitter
        emitter = typing.cast(DummyAsyncEmitter, self.worker.emitter_instance)
        emitter.pause_evt.set()
        # Simulate worker being started, so controller exists
        self.worker.controller = self.mock_controller

        cmd = {"command": "resume"}
        result = self.worker._handle_resume(cmd, self.mock_conn)

        assert result  # resume should return True to continue
        assert not emitter.pause_evt.is_set()  # Check via emitter
        self.mock_conn.send_message.assert_called_with(
            "status", "resumed", status="running"
        )  # Ensure status kwarg
        self.mock_controller.resume.assert_called_once()

    def test_command_sequence(self):
        """Test a sequence of commands."""
        # Initially
        assert not self.worker._running
        emitter = typing.cast(DummyAsyncEmitter, self.worker.emitter_instance)
        assert not emitter.pause_evt.is_set()

        # Simulate worker being started for pause/resume to make sense via controller
        # The 'start' command itself would set up the controller in WorkerBase
        # For direct handle_command tests on pause/resume, ensure controller is set
        self.worker.controller = self.mock_controller

        # Pause the worker
        pause_cmd = {"command": "pause"}
        result = self.worker.handle_command(pause_cmd, self.mock_conn)
        assert result
        assert emitter.pause_evt.is_set()
        self.mock_controller.pause.assert_called_once()

        # Resume the worker
        resume_cmd = {"command": "resume"}
        result = self.worker.handle_command(resume_cmd, self.mock_conn)
        assert result
        assert not emitter.pause_evt.is_set()
        self.mock_controller.resume.assert_called_once()

        # Stop the worker
        stop_cmd = {"command": "stop"}
        result = self.worker.handle_command(stop_cmd, self.mock_conn)
        assert not result
        assert not self.worker._running  # Check _running state directly set by _handle_stop
        self.mock_controller.stop.assert_called_once()  # Controller's stop should be called

    def test_multiple_connection_messages(self):
        """Test that commands send appropriate messages."""
        # Commands that are expected to succeed when controller is available
        commands_and_expected_status_with_controller = [
            ({"command": "pause"}, "paused", "paused"),
            ({"command": "resume"}, "resumed", "running"),
        ]

        # Test with controller active
        self.worker.controller = self.mock_controller
        # Ensure the mocked controller's emitter is the actual one from the worker
        self.mock_controller.emitter = self.worker.emitter_instance

        for (
            cmd,
            expected_msg_data,
            expected_status_kwarg,
        ) in commands_and_expected_status_with_controller:
            self.mock_conn.reset_mock()
            self.mock_controller.reset_mock()  # Reset controller mocks too
            emitter = typing.cast(DummyAsyncEmitter, self.worker.emitter_instance)
            if cmd["command"] == "resume":  # Ensure pause_evt is set before resume
                emitter.pause_evt.set()
            else:
                emitter.pause_evt.clear()

            self.worker.handle_command(cmd, self.mock_conn)
            self.mock_conn.send_message.assert_called_once_with(
                "status", expected_msg_data, status=expected_status_kwarg
            )
            if cmd["command"] == "pause":
                self.mock_controller.pause.assert_called_once()
            elif cmd["command"] == "resume":
                self.mock_controller.resume.assert_called_once()

        # Test stop command (controller will be present from above)
        self.mock_conn.reset_mock()
        self.mock_controller.reset_mock()
        stop_cmd = {"command": "stop"}
        self.worker.handle_command(stop_cmd, self.mock_conn)
        self.mock_conn.send_message.assert_called_once_with(
            "status", "stopped", status="stopped"
        )
        self.mock_controller.stop.assert_called_once()

        # Test pause/resume when controller is NOT active (e.g. before 'start')
        self.worker.controller = None  # Explicitly remove controller
        commands_expecting_error_without_controller = [
            ({"command": "pause"}, "Not started", "error"),
            ({"command": "resume"}, "Not started", "error"),
        ]
        for (
            cmd,
            expected_msg_data,
            expected_status_kwarg,
        ) in commands_expecting_error_without_controller:
            self.mock_conn.reset_mock()
            self.worker.handle_command(cmd, self.mock_conn)
            self.mock_conn.send_message.assert_called_once_with(
                "error", expected_msg_data, status=expected_status_kwarg
            )


class TestWorkerBaseExtension:
    """Test extending WorkerBase with custom commands."""

    def test_custom_command_extension(self):
        """Test that WorkerBase can be extended with custom commands."""

        class CustomWorker(WorkerBase):
            def __init__(self):
                super().__init__()
                # Add custom command
                self._commands["custom"] = self._handle_custom
                self.custom_called = False

            def _handle_custom(self, cmd, conn):
                self.custom_called = True
                conn.send_message("status", "custom_executed")
                return True

        worker = CustomWorker()
        mock_conn = Mock(spec=ConnectionContext)

        # Test custom command
        cmd = {"command": "custom"}
        result = worker.handle_command(cmd, mock_conn)

        assert result
        assert worker.custom_called
        mock_conn.send_message.assert_called_with("status", "custom_executed")

        # Test that standard commands still work
        stop_cmd = {"command": "stop"}
        result = worker.handle_command(stop_cmd, mock_conn)
        assert not result

    def test_override_standard_command(self):
        """Test overriding a standard command."""

        class OverrideWorker(WorkerBase):
            def __init__(self):
                super().__init__()
                # Override stop command
                self._commands["stop"] = self._custom_stop
                self.custom_stop_called = False

            def _custom_stop(self, cmd, conn):
                self.custom_stop_called = True
                conn.send_message("status", "custom_stopped")
                return False  # Still exit like normal stop

        worker = OverrideWorker()
        mock_conn = Mock(spec=ConnectionContext)

        # Test overridden stop command
        cmd = {"command": "stop"}
        result = worker.handle_command(cmd, mock_conn)

        assert not result  # Should still return False to exit
        assert worker.custom_stop_called
        mock_conn.send_message.assert_called_with("status", "custom_stopped")

        # Verify the original stop behavior is not called
        # (worker._running should not be set to False by the original handler)
        # Since we overrode it, the custom implementation controls the behavior
