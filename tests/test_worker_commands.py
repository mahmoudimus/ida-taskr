"""
Tests for WorkerBase command handling functionality.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ida_taskr.worker import ConnectionContext, WorkerBase


class TestWorkerCommands(unittest.TestCase):
    """Test suite for WorkerBase command handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.worker = WorkerBase()
        self.mock_conn = Mock(spec=ConnectionContext)

    def test_worker_initialization(self):
        """Test WorkerBase initialization includes command handlers."""
        worker = WorkerBase()

        self.assertIn("stop", worker._commands)
        self.assertIn("pause", worker._commands)
        self.assertIn("resume", worker._commands)
        self.assertFalse(worker._running)
        self.assertFalse(worker._paused)

    def test_handle_command_with_valid_command(self):
        """Test handling valid commands."""
        # Test stop command
        cmd = {"command": "stop"}
        result = self.worker.handle_command(cmd, self.mock_conn)

        self.assertFalse(result)  # stop should return False to exit
        self.assertFalse(self.worker._running)
        self.mock_conn.send_message.assert_called_with("status", "stopped")

    def test_handle_command_with_invalid_command(self):
        """Test handling invalid/unknown commands."""
        cmd = {"command": "unknown_command"}
        result = self.worker.handle_command(cmd, self.mock_conn)

        self.assertTrue(result)  # unknown commands should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_command_with_missing_command(self):
        """Test handling command dict without 'command' key."""
        cmd = {"other_field": "value"}
        result = self.worker.handle_command(cmd, self.mock_conn)

        self.assertTrue(result)  # missing command should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_command_with_none_command(self):
        """Test handling command dict with None command."""
        cmd = {"command": None}
        result = self.worker.handle_command(cmd, self.mock_conn)

        self.assertTrue(result)  # None command should return True to continue
        self.mock_conn.send_message.assert_not_called()

    def test_handle_stop_command(self):
        """Test stop command handling."""
        cmd = {"command": "stop"}
        result = self.worker._handle_stop(cmd, self.mock_conn)

        self.assertFalse(result)  # stop should return False
        self.assertFalse(self.worker._running)
        self.mock_conn.send_message.assert_called_with("status", "stopped")

    def test_handle_pause_command(self):
        """Test pause command handling."""
        cmd = {"command": "pause"}
        result = self.worker._handle_pause(cmd, self.mock_conn)

        self.assertTrue(result)  # pause should return True to continue
        self.assertTrue(self.worker._paused)
        self.mock_conn.send_message.assert_called_with("status", "paused")

    def test_handle_resume_command(self):
        """Test resume command handling."""
        # First pause the worker
        self.worker._paused = True

        cmd = {"command": "resume"}
        result = self.worker._handle_resume(cmd, self.mock_conn)

        self.assertTrue(result)  # resume should return True to continue
        self.assertFalse(self.worker._paused)
        self.mock_conn.send_message.assert_called_with("status", "resumed")

    def test_command_sequence(self):
        """Test a sequence of commands."""
        # Initially both should be False
        self.assertFalse(self.worker._running)
        self.assertFalse(self.worker._paused)

        # Pause the worker
        pause_cmd = {"command": "pause"}
        result = self.worker.handle_command(pause_cmd, self.mock_conn)
        self.assertTrue(result)
        self.assertTrue(self.worker._paused)

        # Resume the worker
        resume_cmd = {"command": "resume"}
        result = self.worker.handle_command(resume_cmd, self.mock_conn)
        self.assertTrue(result)
        self.assertFalse(self.worker._paused)

        # Stop the worker
        stop_cmd = {"command": "stop"}
        result = self.worker.handle_command(stop_cmd, self.mock_conn)
        self.assertFalse(result)
        self.assertFalse(self.worker._running)

    def test_multiple_connection_messages(self):
        """Test that commands send appropriate messages."""
        commands_and_expected_status = [
            ({"command": "pause"}, "paused"),
            ({"command": "resume"}, "resumed"),
            ({"command": "stop"}, "stopped"),
        ]

        for cmd, expected_status in commands_and_expected_status:
            # Reset mock
            self.mock_conn.reset_mock()

            # Execute command
            self.worker.handle_command(cmd, self.mock_conn)

            # Verify correct status message was sent
            self.mock_conn.send_message.assert_called_once_with(
                "status", expected_status
            )


class TestWorkerBaseExtension(unittest.TestCase):
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

        self.assertTrue(result)
        self.assertTrue(worker.custom_called)
        mock_conn.send_message.assert_called_with("status", "custom_executed")

        # Test that standard commands still work
        stop_cmd = {"command": "stop"}
        result = worker.handle_command(stop_cmd, mock_conn)
        self.assertFalse(result)

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

        self.assertFalse(result)  # Should still return False to exit
        self.assertTrue(worker.custom_stop_called)
        mock_conn.send_message.assert_called_with("status", "custom_stopped")

        # Verify the original stop behavior is not called
        # (worker._running should not be set to False by the original handler)
        # Since we overrode it, the custom implementation controls the behavior


if __name__ == "__main__":
    unittest.main()
