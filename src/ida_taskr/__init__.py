"""
IDA Worker Manager - A reusable task worker manager library for IDA Pro.

This library provides a Qt-based multiprocessing framework for running
CPU-intensive tasks outside of IDA's main thread while maintaining
bidirectional communication.
"""

from .helpers import MultiprocessingHelper, get_logger, is_ida
from .launcher import WorkerLauncher
from .protocols import MessageEmitter, WorkerProtocol

# New import for TaskRunner
from .task_runner import TaskRunner
from .utils import DataProcessorCore
from .worker import ConnectionContext, WorkerBase

# QtAsyncio integration (optional)
from .qt_compat import QT_ASYNCIO_AVAILABLE

__all__ = [
    "WorkerLauncher",
    "WorkerBase",
    "ConnectionContext",
    "MultiprocessingHelper",
    "get_logger",
    "is_ida",
    "WorkerProtocol",
    "MessageEmitter",
    "TaskRunner",
    "DataProcessorCore",
    "QT_ASYNCIO_AVAILABLE",
]

# Conditionally export qtasyncio utilities if available
if QT_ASYNCIO_AVAILABLE:
    try:
        from .qtasyncio import (
            # Asyncio integration
            QAsyncioEventLoop,
            QAsyncioEventLoopPolicy,
            run as qtasyncio_run,
            set_event_loop_policy,
            # Thread executor
            ThreadExecutor,
            Task,
            FutureWatcher,
            # Worker utilities
            WorkerBase as QtWorkerBase,
            FunctionWorker,
            GeneratorWorker,
            create_worker,
            thread_worker,
            new_worker_qthread,
        )

        __all__.extend([
            # Asyncio integration
            "QAsyncioEventLoop",
            "QAsyncioEventLoopPolicy",
            "qtasyncio_run",
            "set_event_loop_policy",
            # Thread executor
            "ThreadExecutor",
            "Task",
            "FutureWatcher",
            # Worker utilities
            "QtWorkerBase",
            "FunctionWorker",
            "GeneratorWorker",
            "create_worker",
            "thread_worker",
            "new_worker_qthread",
        ])
    except ImportError:
        # QtAsyncio module not available
        pass

__version__ = "1.0.0"
