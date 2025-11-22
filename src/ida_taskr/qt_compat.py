"""Qt compatibility layer for PyQt5 and PySide6.

This module provides a unified API for Qt signals and slots across different
Qt bindings used by IDA Pro and standalone environments.
"""

# Try to import Qt frameworks in order of preference
QT_API = None
QT_AVAILABLE = False
QtCore = None
Signal = None
Slot = None
QT_VERSION = None
QProcessEnvironment = None

# Try PySide6 first (IDA Pro 9.2+, modern official Qt bindings)
try:
    from PySide6 import QtCore
    from PySide6.QtCore import Signal, Slot

    QT_API = "PySide6"
    QT_VERSION = QtCore.__version__
    QProcessEnvironment = QtCore.QProcessEnvironment
    QT_AVAILABLE = True
except (ImportError, NotImplementedError):
    pass

# Try PyQt5 (IDA Pro 9.1 and earlier)
if QT_API is None:
    try:
        from PyQt5 import QtCore
        from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

        QT_API = "PyQt5"
        QT_VERSION = QtCore.PYQT_VERSION_STR
        QProcessEnvironment = QtCore.QProcessEnvironment
        QT_AVAILABLE = True
    except (ImportError, NotImplementedError):
        pass

# If no Qt found, create mock classes for import compatibility
if QT_API is None:

    class QtCore:  # type: ignore
        """Mock QtCore module when Qt is not available."""

        class QThread:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Qt is not available. Cannot use WorkerLauncher without Qt."
                )

        class QObject:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Qt is not available. Cannot use WorkerLauncher without Qt."
                )

        class QProcess:
            # Process error enum values (mock)
            class ProcessError:
                FailedToStart = 0
                Crashed = 1
                Timedout = 2
                WriteError = 4
                ReadError = 3
                UnknownError = 5

            # Process state enum values (mock)
            class ProcessState:
                NotRunning = 0
                Starting = 1
                Running = 2

            # Direct enum access (compatibility)
            FailedToStart = 0
            Crashed = 1
            Timedout = 2
            WriteError = 4
            ReadError = 3
            UnknownError = 5
            NotRunning = 0
            Starting = 1
            Running = 2
            NormalExit = 0
            CrashExit = 1

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Qt is not available. Cannot use WorkerLauncher without Qt."
                )

        class QSocketNotifier:
            Read = 1
            Write = 2

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Qt is not available. Cannot use WorkerLauncher without Qt."
                )

    # Dummy signal for type hints
    Signal = lambda *args: None  # type: ignore
    Slot = lambda *args: None  # type: ignore

    class QProcessEnvironment:  # type: ignore
        @staticmethod
        def systemEnvironment():
            raise ImportError(
                "Qt is not available. Cannot use WorkerLauncher without Qt."
            )


def get_qt_api():
    """Return the name of the Qt API being used, or None if Qt is not available."""
    return QT_API


def get_qt_version():
    """Return the version of the Qt framework being used, or None if Qt is not available."""
    return QT_VERSION


# QtAsyncio module availability
QT_ASYNCIO_AVAILABLE = False
qtasyncio = None

if QT_AVAILABLE:
    try:
        from . import qtasyncio
        QT_ASYNCIO_AVAILABLE = True
    except ImportError:
        pass


__all__ = [
    "QtCore",
    "Signal",
    "Slot",
    "QT_API",
    "QT_VERSION",
    "QT_AVAILABLE",
    "QProcessEnvironment",
    "get_qt_api",
    "get_qt_version",
    "QT_ASYNCIO_AVAILABLE",
    "qtasyncio",
]
