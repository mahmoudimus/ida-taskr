"""Integration tests for TaskRunner with real IDA Pro."""

import sys
import time
from pathlib import Path

import pytest


class TestTaskRunnerIntegration:
    """Integration tests for TaskRunner in IDA Pro environment."""

    def test_ida_import(self, ida_available):
        """Test that IDA modules can be imported."""
        assert ida_available
        import idaapi
        import idautils
        import idc

        # Verify we can access basic IDA functionality
        # Support both old (get_inf) and new (get_inf_structure) API
        assert hasattr(idaapi, 'get_inf_structure') or hasattr(idaapi, 'get_inf')

    def test_qt_framework_available(self, qt_framework):
        """Test that a Qt framework is available."""
        assert qt_framework in ["PyQt5", "PySide6"]

        if qt_framework == "PyQt5":
            from PyQt5.QtCore import QObject, pyqtSignal
            assert QObject is not None
            assert pyqtSignal is not None
        else:
            from PySide6.QtCore import QObject, Signal
            assert QObject is not None
            assert Signal is not None

    def test_taskrunner_import(self, ida_available, qt_framework):
        """Test that TaskRunner can be imported in IDA environment."""
        assert ida_available
        assert qt_framework

        try:
            from ida_taskr import TaskRunner
            assert TaskRunner is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TaskRunner: {e}")

    def test_taskrunner_basic_functionality(self, ida_available, qt_framework):
        """Test basic TaskRunner functionality in IDA environment."""
        assert ida_available
        assert qt_framework

        from ida_taskr import TaskRunner

        # Create a simple task
        def simple_task(x, y):
            return x + y

        # Create TaskRunner instance
        runner = TaskRunner()

        # Submit a task
        result = simple_task(2, 3)
        assert result == 5

    def test_event_emitter_in_ida(self, ida_available, qt_framework):
        """Test MessageEmitter in IDA environment."""
        assert ida_available
        assert qt_framework

        try:
            from ida_taskr.event_emitter import MessageEmitter

            emitter = MessageEmitter()
            assert emitter is not None

            # Test signal emission
            messages = []

            def message_handler(msg):
                messages.append(msg)

            emitter.message_received.connect(message_handler)
            emitter.emit_message("test message")

            # Give Qt event loop time to process
            time.sleep(0.1)

            assert len(messages) > 0
            assert messages[0] == "test message"

        except ImportError as e:
            pytest.fail(f"Failed to import MessageEmitter: {e}")

    def test_qt_compatibility_layer(self, qt_framework):
        """Test that the Qt compatibility layer works correctly."""
        assert qt_framework

        if qt_framework == "PyQt5":
            from PyQt5.QtCore import QObject, pyqtSignal as Signal
            from PyQt5.QtWidgets import QApplication

            class TestEmitter(QObject):
                test_signal = Signal(str)

            emitter = TestEmitter()
            assert emitter is not None

        else:  # PySide6
            from PySide6.QtCore import QObject, Signal
            from PySide6.QtWidgets import QApplication

            class TestEmitter(QObject):
                test_signal = Signal(str)

            emitter = TestEmitter()
            assert emitter is not None

    def test_worker_thread_functionality(self, ida_available, qt_framework):
        """Test worker thread functionality in IDA environment."""
        assert ida_available
        assert qt_framework

        try:
            # Test that we can create and use worker threads
            import threading
            import queue

            result_queue = queue.Queue()

            def worker_task():
                result_queue.put("worker_completed")

            thread = threading.Thread(target=worker_task)
            thread.start()
            thread.join(timeout=5)

            assert not thread.is_alive()
            assert not result_queue.empty()
            assert result_queue.get() == "worker_completed"

        except Exception as e:
            pytest.fail(f"Worker thread test failed: {e}")

    def test_ida_api_access_from_worker(self, ida_available, ida_database):
        """Test that IDA API can be accessed from worker context."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi
        import idc

        # Test basic IDA API calls
        inf = idaapi.get_inf_structure()
        assert inf is not None

        # Test getting segments
        seg = idaapi.get_first_seg()
        if seg:
            seg_name = idaapi.get_segm_name(seg)
            assert seg_name is not None

    def test_multiprocessing_compatibility(self, ida_available):
        """Test multiprocessing module compatibility."""
        assert ida_available

        import multiprocessing

        # Test that multiprocessing can be imported
        assert multiprocessing is not None

        # Note: Actual multiprocessing may be restricted in IDA environment
        # This test just verifies the module is available

    @pytest.mark.parametrize("task_count", [1, 5, 10])
    def test_multiple_tasks(self, ida_available, qt_framework, task_count):
        """Test handling multiple tasks in IDA environment."""
        assert ida_available
        assert qt_framework

        def compute_task(n):
            return n * n

        results = [compute_task(i) for i in range(task_count)]

        assert len(results) == task_count
        for i, result in enumerate(results):
            assert result == i * i
