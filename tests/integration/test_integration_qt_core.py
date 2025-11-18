"""Integration tests for Qt Core functionality without IDA Pro requirement.

These tests verify that ida-taskr's Qt-based process management and threading
work correctly in headless mode with both PyQt5 and PySide6.
"""

import sys
import time
import tempfile
from pathlib import Path

import pytest


class TestQtCoreFramework:
    """Test Qt Core framework availability and basic functionality."""

    def test_qt_framework_import(self, qt_framework_headless):
        """Test that Qt framework can be imported in headless mode."""
        assert qt_framework_headless in ["PyQt5", "PySide6"]

        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QObject, QThread, QProcess, pyqtSignal
            assert QObject is not None
            assert QThread is not None
            assert QProcess is not None
            assert pyqtSignal is not None
        else:
            from PySide6.QtCore import QObject, QThread, QProcess, Signal
            assert QObject is not None
            assert QThread is not None
            assert QProcess is not None
            assert Signal is not None

    def test_qprocess_available(self, qt_framework_headless, qtbot):
        """Test that QProcess is available for process management."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcess
        else:
            from PySide6.QtCore import QProcess

        # Create a simple QProcess
        process = QProcess()
        assert process is not None
        assert process.state() == QProcess.NotRunning

    def test_qthread_available(self, qt_framework_headless, qtbot):
        """Test that QThread is available for threading."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal
        else:
            from PySide6.QtCore import QThread, QObject, Signal

        class TestThread(QThread):
            finished_signal = Signal()

            def run(self):
                self.finished_signal.emit()

        thread = TestThread()
        assert thread is not None
        assert not thread.isRunning()

    def test_signal_slot_mechanism(self, qt_framework_headless, qtbot):
        """Test Qt signal/slot mechanism in headless mode."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QObject, pyqtSignal as Signal
        else:
            from PySide6.QtCore import QObject, Signal

        class Emitter(QObject):
            test_signal = Signal(str)

        emitter = Emitter()
        received = []

        def handler(msg):
            received.append(msg)

        emitter.test_signal.connect(handler)
        emitter.test_signal.emit("test_message")

        # Process Qt event loop
        qtbot.wait(10)

        assert len(received) == 1
        assert received[0] == "test_message"


class TestMessageEmitter:
    """Test MessageEmitter with real Qt signals."""

    def test_message_emitter_import(self, qt_framework_headless):
        """Test that MessageEmitter can be imported."""
        from ida_taskr.event_emitter import MessageEmitter
        assert MessageEmitter is not None

    def test_message_emitter_creation(self, qt_framework_headless, qtbot):
        """Test MessageEmitter instance creation."""
        from ida_taskr.event_emitter import MessageEmitter

        emitter = MessageEmitter()
        assert emitter is not None

    def test_message_emitter_signals(self, qt_framework_headless, qtbot):
        """Test MessageEmitter signal emission and reception."""
        from ida_taskr.event_emitter import MessageEmitter

        emitter = MessageEmitter()
        received_messages = []

        def message_handler(msg):
            received_messages.append(msg)

        emitter.message_received.connect(message_handler)
        emitter.emit_message("test_message")

        # Wait for signal processing
        qtbot.wait(10)

        assert len(received_messages) == 1
        assert received_messages[0] == "test_message"

    def test_message_emitter_progress(self, qt_framework_headless, qtbot):
        """Test MessageEmitter progress signal."""
        from ida_taskr.event_emitter import MessageEmitter

        emitter = MessageEmitter()
        progress_updates = []

        def progress_handler(current, total, message):
            progress_updates.append((current, total, message))

        emitter.progress_updated.connect(progress_handler)
        emitter.emit_progress(50, 100, "halfway")

        qtbot.wait(10)

        assert len(progress_updates) == 1
        assert progress_updates[0] == (50, 100, "halfway")

    def test_message_emitter_results(self, qt_framework_headless, qtbot):
        """Test MessageEmitter results signal."""
        from ida_taskr.event_emitter import MessageEmitter

        emitter = MessageEmitter()
        results = []

        def results_handler(result):
            results.append(result)

        emitter.results_ready.connect(results_handler)
        emitter.emit_results({"data": "test_result"})

        qtbot.wait(10)

        assert len(results) == 1
        assert results[0] == {"data": "test_result"}


class TestQProcessBasics:
    """Test QProcess basic functionality for worker launching."""

    def test_qprocess_simple_execution(self, qt_framework_headless, qtbot):
        """Test QProcess can execute a simple command."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcess
        else:
            from PySide6.QtCore import QProcess

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

    def test_qprocess_output_capture(self, qt_framework_headless, qtbot):
        """Test QProcess can capture output from subprocess."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcess
        else:
            from PySide6.QtCore import QProcess

        process = QProcess()
        process.start("python3", ["-c", "print('hello world')"])

        assert process.waitForStarted(5000)
        assert process.waitForFinished(5000)

        output = process.readAllStandardOutput().data().decode('utf-8').strip()
        assert "hello world" in output

    def test_qprocess_error_detection(self, qt_framework_headless, qtbot):
        """Test QProcess error detection for invalid command."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcess
        else:
            from PySide6.QtCore import QProcess

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

    def test_worker_launcher_import(self, qt_framework_headless):
        """Test that WorkerLauncher can be imported."""
        from ida_taskr.launcher import WorkerLauncher
        assert WorkerLauncher is not None

    def test_worker_launcher_qprocess_inheritance(self, qt_framework_headless):
        """Test that WorkerLauncher inherits from QProcess."""
        from ida_taskr.launcher import WorkerLauncher

        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcess
        else:
            from PySide6.QtCore import QProcess

        assert issubclass(WorkerLauncher, QProcess)

    def test_connection_reader_qthread(self, qt_framework_headless):
        """Test that ConnectionReader is a QThread."""
        from ida_taskr.launcher import ConnectionReader

        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QThread
        else:
            from PySide6.QtCore import QThread

        assert issubclass(ConnectionReader, QThread)

    def test_qt_listener_qobject(self, qt_framework_headless):
        """Test that QtListener is a QObject."""
        from ida_taskr.launcher import QtListener

        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QObject
        else:
            from PySide6.QtCore import QObject

        assert issubclass(QtListener, QObject)


class TestTaskRunnerQtIntegration:
    """Test TaskRunner Qt integration in headless mode."""

    def test_taskrunner_import(self, qt_framework_headless):
        """Test that TaskRunner can be imported with Qt available."""
        from ida_taskr import TaskRunner
        assert TaskRunner is not None

    def test_taskrunner_creation_with_qt(self, qt_framework_headless, qtbot):
        """Test TaskRunner instance creation with Qt framework."""
        from ida_taskr import TaskRunner

        # Create TaskRunner instance
        runner = TaskRunner()
        assert runner is not None
        assert hasattr(runner, 'launcher')
        assert hasattr(runner, 'message_emitter')

    def test_taskrunner_message_emitter_type(self, qt_framework_headless, qtbot):
        """Test that TaskRunner uses MessageEmitter."""
        from ida_taskr import TaskRunner
        from ida_taskr.event_emitter import MessageEmitter

        runner = TaskRunner()
        assert isinstance(runner.message_emitter, MessageEmitter)

    def test_taskrunner_launcher_type(self, qt_framework_headless, qtbot):
        """Test that TaskRunner uses WorkerLauncher."""
        from ida_taskr import TaskRunner
        from ida_taskr.launcher import WorkerLauncher

        runner = TaskRunner()
        assert isinstance(runner.launcher, WorkerLauncher)


class TestQProcessEnvironment:
    """Test QProcessEnvironment functionality."""

    def test_process_environment_creation(self, qt_framework_headless):
        """Test QProcessEnvironment can be created."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcessEnvironment
        else:
            from PySide6.QtCore import QProcessEnvironment

        env = QProcessEnvironment.systemEnvironment()
        assert env is not None

    def test_process_environment_variables(self, qt_framework_headless):
        """Test QProcessEnvironment can access environment variables."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcessEnvironment
        else:
            from PySide6.QtCore import QProcessEnvironment

        env = QProcessEnvironment.systemEnvironment()

        # PATH should exist in any environment
        assert env.contains("PATH") or env.contains("Path")

    def test_process_environment_insert(self, qt_framework_headless):
        """Test QProcessEnvironment can insert new variables."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QProcessEnvironment
        else:
            from PySide6.QtCore import QProcessEnvironment

        env = QProcessEnvironment.systemEnvironment()
        env.insert("TEST_VAR", "test_value")

        assert env.contains("TEST_VAR")
        assert env.value("TEST_VAR") == "test_value"


class TestQtSignalsAdvanced:
    """Test advanced Qt signal/slot patterns used by ida-taskr."""

    def test_cross_thread_signals(self, qt_framework_headless, qtbot):
        """Test signals can be emitted across thread boundaries."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal
        else:
            from PySide6.QtCore import QThread, QObject, Signal

        class Worker(QThread):
            finished_with_data = Signal(str)

            def run(self):
                self.finished_with_data.emit("thread_completed")

        worker = Worker()
        received = []

        def handler(msg):
            received.append(msg)

        worker.finished_with_data.connect(handler)
        worker.start()

        # Wait for thread to complete
        worker.wait(5000)
        qtbot.wait(10)

        assert len(received) == 1
        assert received[0] == "thread_completed"

    def test_multiple_signal_handlers(self, qt_framework_headless, qtbot):
        """Test multiple handlers can be connected to the same signal."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QObject, pyqtSignal as Signal
        else:
            from PySide6.QtCore import QObject, Signal

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

        qtbot.wait(10)

        assert results["handler1"] == [42]
        assert results["handler2"] == [84]

    def test_signal_disconnection(self, qt_framework_headless, qtbot):
        """Test signal handlers can be disconnected."""
        if qt_framework_headless == "PyQt5":
            from PyQt5.QtCore import QObject, pyqtSignal as Signal
        else:
            from PySide6.QtCore import QObject, Signal

        class Emitter(QObject):
            data_signal = Signal(str)

        emitter = Emitter()
        received = []

        def handler(msg):
            received.append(msg)

        emitter.data_signal.connect(handler)
        emitter.data_signal.emit("first")

        qtbot.wait(10)

        emitter.data_signal.disconnect(handler)
        emitter.data_signal.emit("second")

        qtbot.wait(10)

        # Only first message should be received
        assert len(received) == 1
        assert received[0] == "first"
