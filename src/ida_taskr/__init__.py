"""
IDA Worker Manager - A reusable task worker manager library for IDA Pro.

This library provides a Qt-based multiprocessing framework for running
CPU-intensive tasks outside of IDA's main thread while maintaining
bidirectional communication.
"""

from .helpers import MultiprocessingHelper, get_logger, is_ida
from .launcher import WorkerLauncher
from .protocols import MessageEmitter, WorkerProtocol
from .utils import DataProcessorCore
from .worker import ConnectionContext, WorkerBase

__all__ = [
    "WorkerLauncher",
    "WorkerBase",
    "ConnectionContext",
    "MultiprocessingHelper",
    "get_logger",
    "is_ida",
    "WorkerProtocol",
    "MessageEmitter",
    "DataProcessorCore",
]

__version__ = "1.0.0"
