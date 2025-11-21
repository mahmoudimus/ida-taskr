"""
Tests for TaskRunner functionality.
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ida_taskr.qt_compat import QT_AVAILABLE
from ida_taskr import TaskRunner, get_logger

# Skip all tests if Qt is not available
pytestmark = pytest.mark.skipif(
    not QT_AVAILABLE,
    reason="TaskRunner requires Qt"
)


class TestTaskRunner(unittest.TestCase):
    """Test suite for TaskRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.worker_script = "test_worker.py"
        self.worker_args = {"data_size": 1024, "start_ea": "0x1000", "is64": "1"}

    def test_task_runner_initialization(self):
        """Test TaskRunner initialization with default parameters."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        self.assertEqual(runner.worker_script, self.worker_script)
        self.assertEqual(runner.worker_args, self.worker_args)
        self.assertIsNotNone(runner.logger)
        self.assertIsNotNone(runner.message_emitter)
        self.assertIsNotNone(runner.launcher)
        self.assertIsNone(runner._results_callback)
        self.assertIsNone(runner._progress_callback)

    def test_task_runner_initialization_with_log_level(self):
        """Test TaskRunner initialization with custom log level."""
        runner = TaskRunner(
            self.worker_script, self.worker_args, log_level=logging.DEBUG
        )

        self.assertEqual(runner.worker_script, self.worker_script)
        self.assertEqual(runner.worker_args, self.worker_args)
        self.assertIsNotNone(runner.logger)

    def test_task_runner_initialization_with_custom_logger(self):
        """Test TaskRunner initialization with custom logger."""
        custom_logger = get_logger("test_logger")
        runner = TaskRunner(self.worker_script, self.worker_args, logger=custom_logger)

        self.assertEqual(runner.logger, custom_logger)

    def test_on_results_callback_registration(self):
        """Test registering results callback."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback function
        callback = Mock()

        # Register callback
        runner.on_results(callback)

        self.assertEqual(runner._results_callback, callback)

        # Verify message emitter listener was registered
        # This is indirect testing since we can't easily inspect the emitter's listeners
        self.assertIsNotNone(runner._results_callback)

    def test_on_progress_callback_registration(self):
        """Test registering progress callback."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback function
        callback = Mock()

        # Register callback
        runner.on_progress(callback)

        self.assertEqual(runner._progress_callback, callback)

    @patch("ida_taskr.task_runner.WorkerLauncher")
    def test_start_successful_launch(self, mock_launcher_class):
        """Test successful worker launch."""
        # Setup mock
        mock_launcher = Mock()
        mock_launcher_class.return_value = mock_launcher
        mock_launcher.launch_worker.return_value = True

        runner = TaskRunner(self.worker_script, self.worker_args)

        # Start the runner
        runner.start()

        # Verify launcher was called with correct arguments
        mock_launcher.launch_worker.assert_called_once_with(
            self.worker_script, self.worker_args
        )

    @patch("ida_taskr.task_runner.WorkerLauncher")
    def test_start_failed_launch(self, mock_launcher_class):
        """Test failed worker launch."""
        # Setup mock
        mock_launcher = Mock()
        mock_launcher_class.return_value = mock_launcher
        mock_launcher.launch_worker.return_value = False

        runner = TaskRunner(self.worker_script, self.worker_args)

        # Start the runner
        runner.start()

        # Verify launcher was called
        mock_launcher.launch_worker.assert_called_once_with(
            self.worker_script, self.worker_args
        )

    def test_handle_results_with_callback(self):
        """Test results handling when callback is registered."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback
        callback = Mock()
        runner.on_results(callback)

        # Test results data
        test_results = {"status": "success", "results": [1, 2, 3]}

        # Call the internal handler directly
        runner._handle_results(test_results)

        # Verify callback was called with correct data
        callback.assert_called_once_with(test_results)

    def test_handle_results_without_callback(self):
        """Test results handling when no callback is registered."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Test results data
        test_results = {"status": "success", "results": [1, 2, 3]}

        # Call the internal handler directly (should not raise an exception)
        runner._handle_results(test_results)

    def test_handle_progress_with_callback(self):
        """Test progress handling when callback is registered."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback
        callback = Mock()
        runner.on_progress(callback)

        # Test progress message
        test_message = {"type": "progress", "progress": 0.5, "status": "processing"}

        # Call the internal handler directly
        runner._handle_progress(test_message)

        # Verify callback was called with correct data
        callback.assert_called_once_with(0.5, "processing")

    def test_handle_progress_with_default_values(self):
        """Test progress handling with missing values."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback
        callback = Mock()
        runner.on_progress(callback)

        # Test progress message with missing values
        test_message = {"type": "progress"}

        # Call the internal handler directly
        runner._handle_progress(test_message)

        # Verify callback was called with default values
        callback.assert_called_once_with(0, "unknown")

    def test_handle_progress_non_progress_message(self):
        """Test progress handling with non-progress message."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callback
        callback = Mock()
        runner.on_progress(callback)

        # Test non-progress message
        test_message = {"type": "error", "message": "Something went wrong"}

        # Call the internal handler directly
        runner._handle_progress(test_message)

        # Verify callback was NOT called
        callback.assert_not_called()

    def test_handle_progress_without_callback(self):
        """Test progress handling when no callback is registered."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Test progress message
        test_message = {"type": "progress", "progress": 0.75, "status": "almost done"}

        # Call the internal handler directly (should not raise an exception)
        runner._handle_progress(test_message)

    def test_integration_callbacks(self):
        """Test integration of callbacks with message emitter."""
        runner = TaskRunner(self.worker_script, self.worker_args)

        # Mock callbacks
        results_callback = Mock()
        progress_callback = Mock()

        # Register callbacks
        runner.on_results(results_callback)
        runner.on_progress(progress_callback)

        # Simulate message emitter events
        test_results = {"status": "success", "data": "test"}
        test_progress = {"type": "progress", "progress": 0.8, "status": "working"}

        # Manually trigger the emitter events to test the integration
        runner.message_emitter.emit("worker_results", test_results)
        runner.message_emitter.emit("worker_message", test_progress)

        # Verify callbacks were called
        results_callback.assert_called_once_with(test_results)
        progress_callback.assert_called_once_with(0.8, "working")


class TestTaskRunnerDoctest(unittest.TestCase):
    """Test TaskRunner with simple usage patterns similar to documentation."""

    def test_simple_usage_pattern(self):
        """Test the simple usage pattern from the documentation example."""
        # Mock the launcher to avoid actual worker launching
        with patch("ida_taskr.task_runner.WorkerLauncher") as mock_launcher_class:
            mock_launcher = Mock()
            mock_launcher_class.return_value = mock_launcher
            mock_launcher.launch_worker.return_value = True

            # Track callback invocations
            results_received = []
            progress_received = []

            def on_results(results):
                results_received.append(results)

            def on_progress(progress, status):
                progress_received.append((progress, status))

            # Create and configure runner
            runner = TaskRunner(
                worker_script="path/to/worker.py",
                worker_args={"data_size": 1024, "start_ea": "0x1000", "is64": "1"},
                log_level=logging.DEBUG,
            )
            runner.on_results(on_results)
            runner.on_progress(on_progress)

            # Start the runner
            runner.start()

            # Verify launcher was called
            mock_launcher.launch_worker.assert_called_once()

            # Simulate some events
            runner._handle_results({"status": "success", "results": [1, 2, 3]})
            runner._handle_progress(
                {"type": "progress", "progress": 0.5, "status": "halfway"}
            )

            # Verify callbacks worked
            self.assertEqual(len(results_received), 1)
            self.assertEqual(results_received[0]["status"], "success")
            self.assertEqual(len(progress_received), 1)
            self.assertEqual(progress_received[0], (0.5, "halfway"))


if __name__ == "__main__":
    unittest.main()
