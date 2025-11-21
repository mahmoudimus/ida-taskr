"""
Unit tests for MessageEmitter functionality and event handling patterns.

Tests the composition-based approach for handling worker messages using
the MessageEmitter pattern.
"""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

from ida_taskr.qt_compat import QT_AVAILABLE
from ida_taskr import MessageEmitter, get_logger

# Import WorkerLauncher only if Qt is available
if QT_AVAILABLE:
    from ida_taskr import WorkerLauncher
else:
    WorkerLauncher = None

logger = get_logger(__name__)


class TestMessageEmitter:
    """Test suite for MessageEmitter event handling."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.emitter = MessageEmitter()
        self.test_events = []

    def teardown_method(self):
        """Clean up after each test method."""
        self.test_events.clear()

    def test_decorator_event_registration(self):
        """Test event handler registration using decorator syntax."""

        @self.emitter.on("worker_connected")
        def on_connected():
            self.test_events.append("connected")

        @self.emitter.on("worker_message")
        def on_message(message: dict):
            self.test_events.append(f"message: {message}")

        @self.emitter.on("worker_results")
        def on_results(results: dict):
            self.test_events.append(f"results: {results}")

        @self.emitter.on("worker_error")
        def on_error(error: str):
            self.test_events.append(f"error: {error}")

        @self.emitter.on("worker_disconnected")
        def on_disconnected():
            self.test_events.append("disconnected")

        # Test events are fired correctly
        self.emitter.emit_worker_connected()
        self.emitter.emit_worker_message({"type": "progress", "progress": 0.5})
        self.emitter.emit_worker_results({"status": "success", "results": [1, 2, 3]})
        self.emitter.emit_worker_error("Test error")
        self.emitter.emit_worker_disconnected()

        # Verify all events were captured
        assert len(self.test_events) == 5
        assert "connected" in self.test_events
        assert "message: {'type': 'progress', 'progress': 0.5}" in self.test_events
        assert "results: {'status': 'success', 'results': [1, 2, 3]}" in self.test_events
        assert "error: Test error" in self.test_events
        assert "disconnected" in self.test_events

    def test_direct_event_registration(self):
        """Test alternative way to register handlers directly without decorators."""

        def handle_connection():
            self.test_events.append("worker_connected")

        def handle_message(message: dict):
            self.test_events.append(f"worker_message: {message}")

        def handle_results(results: dict):
            self.test_events.append(f"worker_results: {results}")

        def handle_error(error: str):
            self.test_events.append(f"worker_error: {error}")

        def handle_disconnection():
            self.test_events.append("worker_disconnected")

        # Register the handlers directly
        self.emitter.on("worker_connected", handle_connection)
        self.emitter.on("worker_message", handle_message)
        self.emitter.on("worker_results", handle_results)
        self.emitter.on("worker_error", handle_error)
        self.emitter.on("worker_disconnected", handle_disconnection)

        # Emit events
        self.emitter.emit_worker_connected()
        test_message = {"type": "status", "data": "processing"}
        self.emitter.emit_worker_message(test_message)
        test_results = {"status": "completed", "results": ["item1", "item2"]}
        self.emitter.emit_worker_results(test_results)
        self.emitter.emit_worker_error("Connection timeout")
        self.emitter.emit_worker_disconnected()

        # Verify events were handled
        assert len(self.test_events) == 5
        assert "worker_connected" in self.test_events
        assert f"worker_message: {test_message}" in self.test_events
        assert f"worker_results: {test_results}" in self.test_events
        assert "worker_error: Connection timeout" in self.test_events
        assert "worker_disconnected" in self.test_events

    def test_multiple_subscribers_same_event(self):
        """Test multiple handlers for the same event."""
        results_log = []

        @self.emitter.on("worker_results")
        def log_results(results: dict):
            results_log.append(f"logged: {len(results.get('results', []))}")

        @self.emitter.on("worker_results")
        def process_results(results: dict):
            results_log.append("processed")

        @self.emitter.on("worker_results")
        def notify_ui(results: dict):
            results_log.append("ui_updated")

        # Emit results event
        test_results = {"status": "success", "results": [1, 2, 3, 4, 5]}
        self.emitter.emit_worker_results(test_results)

        # Verify all handlers were called
        assert len(results_log) == 3
        assert "logged: 5" in results_log
        assert "processed" in results_log
        assert "ui_updated" in results_log

    def test_progress_message_handling(self):
        """Test handling of progress messages specifically."""
        progress_events = []

        @self.emitter.on("worker_message")
        def on_message(message: dict):
            if message.get("type") == "progress":
                progress = message.get("progress", 0)
                progress_events.append(progress * 100)

        # Send progress messages
        self.emitter.emit_worker_message({"type": "progress", "progress": 0.25})
        self.emitter.emit_worker_message({"type": "progress", "progress": 0.50})
        self.emitter.emit_worker_message({"type": "progress", "progress": 0.75})
        self.emitter.emit_worker_message(
            {"type": "status", "message": "working"}
        )  # Should be ignored
        self.emitter.emit_worker_message({"type": "progress", "progress": 1.0})

        # Verify only progress messages were processed
        assert len(progress_events) == 4
        assert progress_events == [25.0, 50.0, 75.0, 100.0]

    def test_error_handling(self):
        """Test error event handling."""
        error_messages = []

        @self.emitter.on("worker_error")
        def on_error(error: str):
            error_messages.append(error)

        # Emit various error types
        self.emitter.emit_worker_error("Connection failed")
        self.emitter.emit_worker_error("Processing timeout")
        self.emitter.emit_worker_error("Invalid data format")

        # Verify all errors were captured
        assert len(error_messages) == 3
        assert "Connection failed" in error_messages
        assert "Processing timeout" in error_messages
        assert "Invalid data format" in error_messages

    @pytest.mark.skip(reason="WorkerLauncher/QProcess requires full Qt event loop, not just QCoreApplication")
    @patch("ida_taskr.launcher.WorkerLauncher.launch_worker")
    def test_worker_launcher_integration(self, mock_launch_worker):
        """Test integration with WorkerLauncher."""
        # Configure the mock
        mock_launch_worker.return_value = True

        # Create message emitter with event handlers
        message_emitter = MessageEmitter()
        connection_events = []

        @message_emitter.on("worker_connected")
        def on_connected():
            connection_events.append("connected")

        @message_emitter.on("worker_disconnected")
        def on_disconnected():
            connection_events.append("disconnected")

        # Create worker launcher with the message emitter
        launcher = WorkerLauncher(message_emitter)

        # Launch worker
        worker_args = {"data_size": 1024, "start_ea": "0x1000", "is64": "1"}
        result = launcher.launch_worker("path/to/test/worker.py", worker_args)

        # Verify launcher was called correctly
        assert result is True
        mock_launch_worker.assert_called_once_with(
            "path/to/test/worker.py", worker_args
        )

        # Test event emission
        message_emitter.emit_worker_connected()
        message_emitter.emit_worker_disconnected()

        assert connection_events == ["connected", "disconnected"]

    def test_results_processing(self):
        """Test processing of worker results."""
        processed_results = []

        @self.emitter.on("worker_results")
        def on_results(results: dict):
            if results.get("status") == "success":
                data = results.get("results", [])
                processed_results.extend(data)

        # Test successful results
        self.emitter.emit_worker_results(
            {"status": "success", "results": ["item1", "item2", "item3"]}
        )

        # Test failed results (should be ignored)
        self.emitter.emit_worker_results(
            {"status": "error", "results": ["should_be_ignored"]}
        )

        # Verify only successful results were processed
        assert processed_results == ["item1", "item2", "item3"]

    def test_no_handlers_registered(self):
        """Test that emitting events with no handlers doesn't cause errors."""
        # Should not raise any exceptions
        self.emitter.emit_worker_connected()
        self.emitter.emit_worker_message({"test": "message"})
        self.emitter.emit_worker_results({"test": "results"})
        self.emitter.emit_worker_error("test error")
        self.emitter.emit_worker_disconnected()

    def test_handler_exceptions_are_propagated(self):
        """Test that exceptions in handlers are propagated (current behavior)."""
        results = []

        @self.emitter.on("worker_message")
        def handler_that_works(message):
            results.append("handler_called")

        @self.emitter.on("worker_message")
        def handler_that_fails(message):
            raise Exception("Handler failed")

        # The current implementation propagates exceptions, so we expect one
        with pytest.raises(Exception) as exc_info:
            self.emitter.emit_worker_message({"test": "message"})

        assert str(exc_info.value) == "Handler failed"

        # Due to the way Python sets work, handler order is not guaranteed.
        # The exception will stop execution, so we can only guarantee the exception was raised.
        # We cannot guarantee which handlers ran before the failing one.
        assert True  # If we got here, the exception was properly propagated
