"""Integration tests for Qt Core functionality without IDA Pro requirement.

These tests verify that ida-taskr's Qt-based process management and threading
work correctly in headless mode with PyQt5, PyQt6, or PySide6.
"""

import sys
import time
import tempfile
from pathlib import Path

import pytest

# Import from qt_compat to get the unified Signal/Slot API
from ida_taskr.qt_compat import QtCore, Signal, QT_API

# Import Qt Core components
QObject = QtCore.QObject
QThread = QtCore.QThread
QProcess = QtCore.QProcess
QProcessEnvironment = QtCore.QProcessEnvironment


class TestQtCoreFramework:
    """Test Qt Core framework availability and basic functionality."""

    def test_qt_framework_import(self):
        """Test that Qt framework can be imported in headless mode."""
        # These imports already happened at module level, just verify they work
        assert QObject is not None
        assert QThread is not None
        assert QProcess is not None
        assert Signal is not None

    def test_qprocess_available(self):
        """Test that QProcess is available for process management."""
        # Create a simple QProcess
        process = QProcess()
        assert process is not None
        assert process.state() == QProcess.NotRunning

    def test_qthread_available(self):
        """Test that QThread is available for threading."""
        class TestThread(QThread):
            finished_signal = Signal()

            def run(self):
                self.finished_signal.emit()

        thread = TestThread()
        assert thread is not None
        assert not thread.isRunning()

    def test_signal_slot_mechanism(self):
        """Test Qt signal/slot mechanism in headless mode."""
        class Emitter(QObject):
            test_signal = Signal(str)

        emitter = Emitter()
        received = []

        def handler(msg):
            received.append(msg)

        emitter.test_signal.connect(handler)
        emitter.test_signal.emit("test_message")

        # Signals are delivered synchronously in same thread
        assert len(received) == 1
        assert received[0] == "test_message"


class TestMessageEmitter:
    """Test MessageEmitter with real Qt signals."""

    def test_message_emitter_import(self):
        """Test that MessageEmitter can be imported."""
        from ida_taskr.protocols import MessageEmitter
        assert MessageEmitter is not None

    def test_message_emitter_creation(self):
        """Test MessageEmitter instance creation."""
        from ida_taskr.protocols import MessageEmitter

        emitter = MessageEmitter()
        assert emitter is not None

    def test_message_emitter_signals(self):
        """Test MessageEmitter signal emission and reception."""
        from ida_taskr.protocols import MessageEmitter

        emitter = MessageEmitter()
        received_messages = []

        @emitter.on('worker_message')
        def message_handler(msg):
            received_messages.append(msg)

        emitter.emit_worker_message("test_message")

        assert len(received_messages) == 1
        assert received_messages[0] == "test_message"

    def test_message_emitter_progress(self):
        """Test MessageEmitter progress-like events."""
        from ida_taskr.protocols import MessageEmitter

        emitter = MessageEmitter()
        received = []

        @emitter.on('worker_message')
        def handler(msg):
            received.append(msg)

        emitter.emit_worker_message({"type": "progress", "current": 50, "total": 100})

        assert len(received) == 1
        assert received[0]["type"] == "progress"

    def test_message_emitter_results(self):
        """Test MessageEmitter results signal."""
        from ida_taskr.protocols import MessageEmitter

        emitter = MessageEmitter()
        results = []

        @emitter.on('worker_results')
        def results_handler(result):
            results.append(result)

        emitter.emit_worker_results({"data": "test_result"})

        assert len(results) == 1
        assert results[0] == {"data": "test_result"}


class TestQProcessBasics:
    """Test QProcess basic functionality for worker launching."""

    def test_qprocess_simple_execution(self):
        """Test QProcess can execute a simple command."""
        process = QProcess()

        # Use waitForStarted/waitForFinished instead of signals for simplicity
        process.start("python3", ["-c", "print('hello')"])

        # Wait for process to start
        started = process.waitForStarted(5000)
        assert started, "Process failed to start"

        # Wait for process to finish
        finished = process.waitForFinished(5000)
        assert finished, "Process failed to finish"

        # Check exit status
        assert process.exitStatus() == QProcess.NormalExit
        assert process.exitCode() == 0

    def test_qprocess_output_capture(self):
        """Test QProcess can capture output from subprocess."""
        process = QProcess()
        process.start("python3", ["-c", "print('hello world')"])

        assert process.waitForStarted(5000)
        assert process.waitForFinished(5000)

        output = process.readAllStandardOutput().data().decode('utf-8').strip()
        assert "hello world" in output

    def test_qprocess_error_detection(self):
        """Test QProcess error detection for invalid command."""
        process = QProcess()

        # Try to start a non-existent command
        process.start("nonexistent_command_12345", [])

        # Wait for error
        started = process.waitForStarted(2000)
        assert not started, "Process should not start with invalid command"

        # Check error occurred
        assert process.error() == QProcess.FailedToStart


class TestWorkerLauncher:
    """Test WorkerLauncher functionality with Qt Core."""

    def test_worker_launcher_import(self):
        """Test that WorkerLauncher can be imported."""
        from ida_taskr.launcher import WorkerLauncher
        assert WorkerLauncher is not None

    def test_worker_launcher_qprocess_inheritance(self):
        """Test that WorkerLauncher inherits from QProcess."""
        from ida_taskr.launcher import WorkerLauncher
        assert issubclass(WorkerLauncher, QProcess)

    def test_connection_reader_qthread(self):
        """Test that ConnectionReader is a QThread."""
        from ida_taskr.launcher import ConnectionReader
        assert issubclass(ConnectionReader, QThread)

    def test_qt_listener_qobject(self):
        """Test that QtListener is a QObject."""
        from ida_taskr.launcher import QtListener
        assert issubclass(QtListener, QObject)


class TestTaskRunnerQtIntegration:
    """Test TaskRunner Qt integration in headless mode."""

    def test_taskrunner_import(self):
        """Test that TaskRunner can be imported with Qt available."""
        from ida_taskr import TaskRunner
        assert TaskRunner is not None

    def test_taskrunner_creation_with_qt(self):
        """Test TaskRunner instance creation with Qt framework."""
        from ida_taskr import TaskRunner

        # Create TaskRunner instance with dummy args
        runner = TaskRunner("dummy_script.py", ["arg1"])
        assert runner is not None
        assert hasattr(runner, 'launcher')
        assert hasattr(runner, 'message_emitter')

    def test_taskrunner_message_emitter_type(self):
        """Test that TaskRunner uses MessageEmitter."""
        from ida_taskr import TaskRunner
        from ida_taskr.protocols import MessageEmitter

        runner = TaskRunner("dummy_script.py", ["arg1"])
        assert isinstance(runner.message_emitter, MessageEmitter)

    def test_taskrunner_launcher_type(self):
        """Test that TaskRunner uses WorkerLauncher."""
        from ida_taskr import TaskRunner
        from ida_taskr.launcher import WorkerLauncher

        runner = TaskRunner("dummy_script.py", ["arg1"])
        assert isinstance(runner.launcher, WorkerLauncher)


class TestQProcessEnvironment:
    """Test QProcessEnvironment functionality."""

    def test_process_environment_creation(self):
        """Test QProcessEnvironment can be created."""
        env = QProcessEnvironment.systemEnvironment()
        assert env is not None

    def test_process_environment_variables(self):
        """Test QProcessEnvironment can access environment variables."""
        env = QProcessEnvironment.systemEnvironment()

        # PATH should exist in any environment
        assert env.contains("PATH") or env.contains("Path")

    def test_process_environment_insert(self):
        """Test QProcessEnvironment can insert new variables."""
        env = QProcessEnvironment.systemEnvironment()
        env.insert("TEST_VAR", "test_value")

        assert env.contains("TEST_VAR")
        assert env.value("TEST_VAR") == "test_value"


class TestQtSignalsAdvanced:
    """Test advanced Qt signal/slot patterns used by ida-taskr."""

    def test_cross_thread_signals(self):
        """Test signals can be emitted across thread boundaries."""
        import time

        class Worker(QThread):
            finished_with_data = Signal(str)

            def run(self):
                time.sleep(0.01)  # Small delay to ensure thread starts
                self.finished_with_data.emit("thread_completed")

        worker = Worker()
        received = []

        def handler(msg):
            received.append(msg)

        worker.finished_with_data.connect(handler)
        worker.start()

        # Wait for thread to complete
        worker.wait(5000)

        # Small delay to let signal propagate (cross-thread signals are queued)
        time.sleep(0.05)

        assert len(received) == 1
        assert received[0] == "thread_completed"

    def test_multiple_signal_handlers(self):
        """Test multiple handlers can be connected to the same signal."""
        class Emitter(QObject):
            data_ready = Signal(int)

        emitter = Emitter()
        results = {"handler1": [], "handler2": []}

        def handler1(value):
            results["handler1"].append(value)

        def handler2(value):
            results["handler2"].append(value * 2)

        emitter.data_ready.connect(handler1)
        emitter.data_ready.connect(handler2)
        emitter.data_ready.emit(42)

        assert results["handler1"] == [42]
        assert results["handler2"] == [84]

    def test_signal_disconnection(self):
        """Test signal handlers can be disconnected."""
        class Emitter(QObject):
            data_signal = Signal(str)

        emitter = Emitter()
        received = []

        def handler(msg):
            received.append(msg)

        emitter.data_signal.connect(handler)
        emitter.data_signal.emit("first")

        emitter.data_signal.disconnect(handler)
        emitter.data_signal.emit("second")

        # Only first message should be received
        assert len(received) == 1
        assert received[0] == "first"
