"""Qt compatibility layer for PyQt5, PyQt6, and PySide6.

This module provides a unified API for Qt signals and slots across different
Qt bindings used by IDA Pro and standalone environments.
"""

import sys

# Try to import Qt frameworks in order of preference
QT_API = None
QtCore = None
Signal = None
Slot = None
QT_VERSION = None

# Try PySide6 first (modern, official Qt bindings)
try:
    from PySide6 import QtCore
    from PySide6.QtCore import Signal, Slot
    QT_API = "PySide6"
    QT_VERSION = QtCore.__version__
except (ImportError, NotImplementedError):
    pass

# Try PyQt6 (modern PyQt bindings)
if QT_API is None:
    try:
        from PyQt6 import QtCore
        from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
        QT_API = "PyQt6"
        QT_VERSION = QtCore.PYQT_VERSION_STR
    except (ImportError, NotImplementedError):
        pass

# Try PyQt5 (IDA Pro 9.1 and earlier)
if QT_API is None:
    try:
        from PyQt5 import QtCore
        from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
        QT_API = "PyQt5"
        QT_VERSION = QtCore.PYQT_VERSION_STR
    except (ImportError, NotImplementedError):
        pass

# If no Qt found, raise an error
if QT_API is None:
    raise ImportError(
        "No Qt framework found. Please install one of: PySide6, PyQt6, or PyQt5"
    )


# PyQt6 compatibility: In PyQt6, enums are nested in classes
# (e.g., QProcess.ProcessState.NotRunning vs QProcess.NotRunning)
# Add compatibility attributes for easier access
if QT_API == "PyQt6":
    # QProcess enums
    if hasattr(QtCore.QProcess, 'ProcessState'):
        QtCore.QProcess.NotRunning = QtCore.QProcess.ProcessState.NotRunning
        QtCore.QProcess.Starting = QtCore.QProcess.ProcessState.Starting
        QtCore.QProcess.Running = QtCore.QProcess.ProcessState.Running

    if hasattr(QtCore.QProcess, 'ProcessError'):
        QtCore.QProcess.FailedToStart = QtCore.QProcess.ProcessError.FailedToStart
        QtCore.QProcess.Crashed = QtCore.QProcess.ProcessError.Crashed
        QtCore.QProcess.Timedout = QtCore.QProcess.ProcessError.Timedout
        QtCore.QProcess.ReadError = QtCore.QProcess.ProcessError.ReadError
        QtCore.QProcess.WriteError = QtCore.QProcess.ProcessError.WriteError
        QtCore.QProcess.UnknownError = QtCore.QProcess.ProcessError.UnknownError

    if hasattr(QtCore.QProcess, 'ExitStatus'):
        QtCore.QProcess.NormalExit = QtCore.QProcess.ExitStatus.NormalExit
        QtCore.QProcess.CrashExit = QtCore.QProcess.ExitStatus.CrashExit


def get_qt_api():
    """Return the name of the Qt API being used."""
    return QT_API


def get_qt_version():
    """Return the version of the Qt framework being used."""
    return QT_VERSION


__all__ = ['QtCore', 'Signal', 'Slot', 'QT_API', 'QT_VERSION', 'get_qt_api', 'get_qt_version']
