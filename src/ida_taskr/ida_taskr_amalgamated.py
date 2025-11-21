"""
ida-taskr - Amalgamated Single-File Version

A Qt-integrated task worker framework for IDA Pro and standalone Python applications.
Combines: helpers, utils, protocols, qt_compat, qtasyncio, worker, launcher, task_runner

Usage:
    from ida_taskr_amalgamated import (
        TaskRunner, WorkerLauncher, WorkerBase, ThreadExecutor,
        ProcessPoolExecutor, InterpreterPoolExecutor, ...
    )
"""

from __future__ import annotations

# =============================================================================
# CONSOLIDATED IMPORTS
# =============================================================================
import asyncio
import atexit
import collections
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import inspect
import logging
import math
import multiprocessing
import multiprocessing.connection
import multiprocessing.shared_memory
import os
import pathlib
import pickle
import select
import stat
import sys
import threading
import time
import typing
import uuid
import warnings
import weakref
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

# =============================================================================
# HELPERS MODULE
# =============================================================================

def is_ida():
    """Check if running inside IDA Pro application."""
    exec_name = pathlib.Path(sys.executable).name.lower()
    return exec_name.startswith(("ida", "idat", "idaw", "idag"))


# Configure stdout encoding
if not is_ida():
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
else:
    sys.stdout.encoding = "utf-8"  # type: ignore


def configure_logging(
    log,
    level=logging.INFO,
    handler_filters=None,
    fmt_str="[%(name)s:%(levelname)s:%(process)d:%(threadName)s] @ %(asctime)s %(message)s",
):
    """Configure logging with proper formatting and filters."""
    log.propagate = False
    log.setLevel(level)
    formatter = logging.Formatter(fmt_str)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    if handler_filters is not None:
        for _filter in handler_filters:
            handler.addFilter(_filter)

    for handler in log.handlers[:]:
        log.removeHandler(handler)
        handler.close()

    if not log.handlers:
        log.addHandler(handler)


def get_logger(name=None, configurer=None, log_level=logging.INFO, custom_logger=None):
    """Get a configured logger instance."""
    if custom_logger:
        return custom_logger
    if not configurer:
        configurer = functools.partial(configure_logging, level=log_level)
    prefix = "ida." if is_ida() else "worker."
    name = name or f"{prefix}ida_taskr"
    logger = logging.getLogger(name)
    configurer(logger)
    return logger


class MultiprocessingHelper:
    """Static helper class for multiprocessing context and Python interpreter discovery."""

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def get_python_interpreter():
        """Gets the path to a suitable Python interpreter."""
        logger = get_logger()

        if (
            hasattr(sys, "_base_executable")
            and sys._base_executable
            and "python" in pathlib.Path(sys._base_executable).name.lower()
        ):
            logger.debug(f"Using _base_executable: {sys._base_executable}")
            return pathlib.Path(sys._base_executable)

        base_paths = [sys.prefix, sys.exec_prefix, sys.executable]
        exe_suffix = ".exe" if os.name == "nt" else ""
        python_name = f"python{exe_suffix}"

        def base_dirs():
            for dirname in map(pathlib.Path, base_paths):
                yield dirname
                yield dirname.parent
                yield dirname.parent.parent

        for dirname in base_dirs():
            for basename in ["", "bin", "python"]:
                interp_path = dirname / basename / python_name
                if not interp_path.exists():
                    continue
                if not (interp_path.is_file() or interp_path.is_symlink()):
                    continue
                st_mode = interp_path.stat().st_mode
                if not st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                    continue
                logger.debug(f"Found Python interpreter at: {interp_path}")
                return interp_path

        logger.warning("Could not determine Python interpreter path, falling back to 'python' in PATH.")
        return pathlib.Path("python")

    @staticmethod
    def set_multiprocessing_context():
        """Sets up the multiprocessing context to use 'spawn' and sets the Python executable."""
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        multiprocessing.set_executable(str(MultiprocessingHelper.get_python_interpreter()))


# Initialize multiprocessing context
MultiprocessingHelper.set_multiprocessing_context()

# Module logger
_logger = get_logger("ida_taskr")


# =============================================================================
# UTILS MODULE
# =============================================================================

def humanize_bytes(
    num_bytes: int, precision: int = 2, units: list[str] = ["B", "KB", "MB", "GB"]
) -> str:
    """Convert a byte count into a human-friendly string with units."""
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if num_bytes == 0:
        return "0 B"
    idx = 0
    value = float(num_bytes)
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if value.is_integer():
        return f"{int(value)} {units[idx]}"
    else:
        return f"{value:.{precision}f} {units[idx]}"


class emit:
    def __init__(self, event):
        self.event = event

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(inst, *args, **kwargs):
            result = fn(inst, *args, **kwargs)
            inst.emit(self.event)
            return result
        return wrapper


T = TypeVar("T")


class reify(Generic[T]):
    """Acts similar to a property, except the result will be set as an attribute."""

    def __init__(self, fn: Callable[..., T]) -> None:
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "<unknown>")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", "") or ""
        self.__qualname__ = getattr(fn, "__qualname__", "") or ""
        self.__annotations__ = getattr(fn, "__annotations__", {})

    @overload
    def __get__(self, instance: None, owner: type) -> "reify[T]": ...
    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...

    def __get__(self, instance: Any, owner: type) -> "T | reify[T]":
        if instance is None:
            return self
        fn = self.fn
        val = fn(instance)
        setattr(instance, fn.__name__, val)
        return val


class EventEmitter:
    @reify
    def _listeners(self):
        return collections.defaultdict(set)

    def on(self, event, handler=None):
        """Register an event handler for the given event."""
        if handler:
            self._listeners[event].add(handler)
            return handler

        @functools.wraps(self.on)
        def decorator(func):
            self.on(event, func)
            return func
        return decorator

    def once(self, event, handler):
        @functools.wraps(handler)
        def once_handler(*args, **kwargs):
            self.remove(event, once_handler)
            return handler(*args, **kwargs)
        self.on(event, once_handler)

    def remove(self, event, handler):
        self._listeners[event].discard(handler)

    def emit(self, event, *args, **kwargs):
        for handler in list(self._listeners[event]):
            handler(*args, **kwargs)


@dataclasses.dataclass
class AsyncEventEmitter(ABC):
    def __post_init__(self):
        self._listeners = collections.defaultdict(set)
        self.pause_evt = asyncio.Event()
        self.stop_evt = asyncio.Event()
        self.logger = get_logger(self.__class__.__name__)

    def on(self, event, handler=None):
        """Register an event handler for the given event."""
        if handler:
            self._listeners[event].add(handler)
            return handler

        @functools.wraps(self.on)
        def decorator(func):
            self.on(event, func)
            return func
        return decorator

    def once(self, event, handler):
        @functools.wraps(handler)
        def once_handler(*args, **kwargs):
            self.remove(event, once_handler)
            return handler(*args, **kwargs)
        self.on(event, once_handler)

    def remove(self, event, handler):
        self._listeners[event].discard(handler)

    async def emit(self, event, *args):
        for handler in list(self._listeners[event]):
            await handler(*args)

    @abstractmethod
    async def run(self):
        """Core asynchronous task execution logic."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Cleanup resources for the async task."""
        pass


def log_execution_time(func, loglvl=logging.INFO):
    """Decorator to log the execution time of async stage methods."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.log(loglvl, f"{func.__qualname__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper


@dataclasses.dataclass
class Range:
    """A range of addresses with a start (inclusive) and end (exclusive)."""
    start: int
    end: int

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start must be less than end")

    def __contains__(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Range") -> bool:
        return self.start < other.end and other.start < self.end

    def merge(self, other: "Range") -> "Range":
        return Range(min(self.start, other.start), max(self.end, other.end))


class IntervalSet:
    """Sorted, non-overlapping list of Range objects with O(log n) insertion."""
    __slots__ = ("_ranges",)

    def __init__(self) -> None:
        self._ranges: list[Range] = []

    def __iter__(self):
        return iter(self._ranges)

    def __len__(self):
        return len(self._ranges)

    def add(self, new: Range) -> None:
        """Insert `new` and coalesce any overlaps / adjacencies in-place."""
        if not self._ranges:
            self._ranges.append(new)
            return

        idx = bisect_left(self._ranges, new.start, key=lambda r: r.start)

        if idx > 0 and self._ranges[idx - 1].end >= new.start:
            idx -= 1

        while idx < len(self._ranges) and new.overlaps(self._ranges[idx]):
            new = new.merge(self._ranges[idx])
            del self._ranges[idx]

        if idx < len(self._ranges) and new.end == self._ranges[idx].start:
            new = new.merge(self._ranges[idx])
            del self._ranges[idx]
        if idx > 0 and self._ranges[idx - 1].end == new.start:
            new = new.merge(self._ranges[idx - 1])
            del self._ranges[idx - 1]
            idx -= 1

        self._ranges.insert(idx, new)

    def covers(self, addr: int) -> bool:
        i = bisect_right(self._ranges, addr, key=lambda r: r.start) - 1
        return i >= 0 and addr < self._ranges[i].end

    def as_tuples(self):
        return [(r.start, r.end) for r in self._ranges]


def resolve_overlaps(ranges: list[Range]) -> IntervalSet:
    """Fast, linear-time overlap resolution."""
    _logger.info(f"Resolving overlaps among {len(ranges)} ranges")
    intervals = IntervalSet()
    for r in ranges:
        intervals.add(r)
        last_end = intervals.as_tuples()[-1][1]
        target = r.end
        if target == last_end:
            _logger.info(f"  Accepted (or widened): {r.start:X}-{r.end:X}")
        else:
            _logger.info(f"  Rejected overlap: {r.start:X}-{r.end:X}")
    return intervals


class PatchManager:
    """Manages deferred patch operations."""

    class Mode(enum.Enum):
        PATCH = enum.auto()
        PUT = enum.auto()

    def __init__(self, patch_mode: Mode = Mode.PATCH, dry_run: bool = False, auto_clear: bool = True):
        self.dry_run = dry_run
        self.patch_mode = patch_mode
        self.pending_patches: list[DeferredPatchOp] = []
        self.auto_clear = auto_clear
        _logger.info("PatchManager initialized (dry_run=%s, mode=%s)", self.dry_run, self.patch_mode.name)

    def add_patch(self, address: int, byte_values: bytes):
        """Creates and queues a DeferredPatchOp."""
        op = DeferredPatchOp(address, byte_values, self.patch_mode)
        self.pending_patches.append(op)
        _logger.debug("Queued patch operation: %s", op)

    def apply_all(self, dry_run_override: bool | None = None) -> bool:
        """Applies all queued patch operations."""
        _logger.info("Applying %d queued patches...", len(self))
        success_count = 0
        fail_count = 0

        if dry_run_override is None:
            dry_run_override = self.dry_run

        for op in self.pending_patches:
            if op.apply(dry_run_override):
                success_count += 1
            else:
                fail_count += 1

        _logger.info("Patch application complete. Success: %d, Failed: %d", success_count, fail_count)
        if self.auto_clear:
            self.pending_patches.clear()
        return fail_count == 0

    def __len__(self) -> int:
        return len(self.pending_patches)


@dataclasses.dataclass(repr=False)
class DeferredPatchOp:
    """Class to store patch operations that will be applied later."""
    address: int
    byte_values: bytes
    mode: PatchManager.Mode
    dry_run: bool = False

    @classmethod
    def patch(cls, address: int, byte_values: bytes, dry_run: bool = False):
        return cls(address, byte_values, PatchManager.Mode.PATCH, dry_run)

    @classmethod
    def put(cls, address: int, byte_values: bytes, dry_run: bool = False):
        return cls(address, byte_values, PatchManager.Mode.PUT, dry_run)

    def apply(self, dry_run_override: bool = False) -> bool:
        """Apply the patch operation."""
        is_dry_run = dry_run_override or self.dry_run
        _logger.info(
            "[*] %sPatching decrypted chunk %s at 0x%X (size: %d)",
            "(Dry Run) " if is_dry_run else "",
            ("revertably" if self.mode == PatchManager.Mode.PATCH else "destructively"),
            self.address,
            len(self.byte_values),
        )
        success = True
        if is_dry_run:
            return success
        try:
            import idaapi
            func = idaapi.put_bytes if self.mode == PatchManager.Mode.PUT else idaapi.patch_bytes
            func(self.address, self.byte_values)
        except Exception as e:
            _logger.error(f"Failed to apply patch {self}: {e}", exc_info=True)
            success = False
        return success

    def __str__(self):
        dry_run_str = " (dry run)" if self.dry_run else ""
        return f"{self.__class__.__name__}({len(self.byte_values)} bytes, mode={self.mode.name}{dry_run_str} @ address=0x{self.address:X})"

    __repr__ = __str__


def make_chunks(buf_len: int, n_chunks: int, max_pat: int):
    """Yield exactly n_chunks tuples of (padded_start, padded_end, core_start, core_end)."""
    for i in range(n_chunks):
        core_start = (buf_len * i) // n_chunks
        core_end = (buf_len * (i + 1)) // n_chunks
        core_len = core_end - core_start
        padded_start = max(0, core_start - (max_pat - 1))
        padded_end = min(buf_len, core_end + (max_pat - 1))
        padded_len = padded_end - padded_start
        _logger.debug(
            "Chunk %2d/%d: core=[%#x-%#x) (%d bytes), padded=[%#x-%#x) (%d bytes)",
            i, n_chunks, core_start, core_end, core_len, padded_start, padded_end, padded_len,
        )
        yield padded_start, padded_end, core_start, core_end


@contextlib.contextmanager
def shm_buffer(name: str, buf_len: int | None = None) -> typing.Generator[
    multiprocessing.shared_memory.SharedMemory | memoryview, None, None
]:
    """Context manager to access the shared memory buffer."""
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    try:
        yield shm.buf[:buf_len] if buf_len else shm
    finally:
        shm.close()


def execute_chunk_with_shm_view(
    user_chunk_processor: typing.Callable[..., typing.Any],
    shm_name: str,
    padded_start: int,
    padded_end: int,
    *user_args: typing.Any,
) -> typing.Any:
    """Handles SHM attachment, memoryview creation, and cleanup for a user's chunk processing function."""
    with shm_buffer(shm_name) as shm_object:
        if not isinstance(shm_object, multiprocessing.shared_memory.SharedMemory):
            _logger.error("shm_buffer did not yield a SharedMemory object as expected. Type: %s", type(shm_object))
            try:
                buffer_to_view = shm_object.buf
            except AttributeError:
                _logger.error("Shared memory object does not have a .buf attribute.")
                raise TypeError("shm_buffer yielded an unexpected object type without a .buf attribute.") from None
        else:
            buffer_to_view = shm_object.buf

        chunk_mv = memoryview(buffer_to_view)[padded_start:padded_end]
        try:
            result = user_chunk_processor(chunk_mv, *user_args)
            return result
        finally:
            del chunk_mv
            _logger.debug("Deleted memoryview for chunk [%d:%d] from shm '%s'", padded_start, padded_end, shm_name)


# =============================================================================
# PROTOCOLS MODULE
# =============================================================================

class WorkerProtocol(ABC):
    """Protocol for worker implementations."""

    @abstractmethod
    def setup(self, **kwargs):
        """Set up the worker."""
        pass

    @abstractmethod
    def process(self, connection, **kwargs):
        """Process tasks."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class MessageEmitter(EventEmitter):
    """Event emitter for handling messages from workers in IDA."""

    def emit_worker_connected(self):
        self.emit("worker_connected")

    def emit_worker_message(self, message: dict):
        self.emit("worker_message", message)

    def emit_worker_results(self, results: dict):
        self.emit("worker_results", results)

    def emit_worker_error(self, error: str):
        self.emit("worker_error", error)

    def emit_worker_disconnected(self):
        self.emit("worker_disconnected")


# =============================================================================
# QT COMPATIBILITY MODULE
# =============================================================================

QT_API = None
QT_AVAILABLE = False
QtCore = None
Signal = None
Slot = None
QT_VERSION = None
QProcessEnvironment = None

# Try PySide6 first
try:
    from PySide6 import QtCore
    from PySide6.QtCore import Signal, Slot
    QT_API = "PySide6"
    QT_VERSION = QtCore.__version__
    QProcessEnvironment = QtCore.QProcessEnvironment
    QT_AVAILABLE = True
except (ImportError, NotImplementedError):
    pass

# Try PyQt5
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

# Mock classes if no Qt found
if QT_API is None:
    class QtCore:  # type: ignore
        class QThread:
            def __init__(self, *args, **kwargs):
                raise ImportError("Qt is not available.")

        class QObject:
            def __init__(self, *args, **kwargs):
                raise ImportError("Qt is not available.")

        class QProcess:
            class ProcessError:
                FailedToStart = 0
                Crashed = 1
                Timedout = 2
                WriteError = 4
                ReadError = 3
                UnknownError = 5

            class ProcessState:
                NotRunning = 0
                Starting = 1
                Running = 2

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
                raise ImportError("Qt is not available.")

        class QSocketNotifier:
            Read = 1
            Write = 2
            def __init__(self, *args, **kwargs):
                raise ImportError("Qt is not available.")

    Signal = lambda *args: None  # type: ignore
    Slot = lambda *args: None  # type: ignore

    class QProcessEnvironment:  # type: ignore
        @staticmethod
        def systemEnvironment():
            raise ImportError("Qt is not available.")


def get_qt_api():
    """Return the name of the Qt API being used."""
    return QT_API


def get_qt_version():
    """Return the version of the Qt framework being used."""
    return QT_VERSION


# =============================================================================
# QTASYNCIO MODULE
# =============================================================================

QT_ASYNCIO_AVAILABLE = False

if QT_AVAILABLE:
    # Import Qt components
    QCoreApplication = QtCore.QCoreApplication
    QEvent = QtCore.QEvent
    QEventLoop = QtCore.QEventLoop
    QMetaObject = QtCore.QMetaObject
    QObject = QtCore.QObject
    QSemaphore = QtCore.QSemaphore
    QThread = QtCore.QThread
    QThreadPool = QtCore.QThreadPool
    QTimer = QtCore.QTimer
    Qt = QtCore.Qt

    try:
        from PySide6.QtCore import Q_ARG, QRunnable
    except ImportError:
        try:
            from PyQt5.QtCore import Q_ARG, QRunnable
        except ImportError:
            Q_ARG = None
            QRunnable = object  # type: ignore

    Future = concurrent.futures.Future

    # --------------------------------------------------------------------------- #
    # Asyncio integration
    # --------------------------------------------------------------------------- #
    class QAsyncioEventLoop(QEventLoop):
        def __init__(self, asyncio_loop: asyncio.AbstractEventLoop, parent: Optional[QObject] = None):
            super().__init__(parent)
            self._asyncio_loop = asyncio_loop

        def processEvents(self, flags=None):
            if flags is None:
                flags = QEventLoop.AllEvents
            self._asyncio_loop.run_until_complete(self._process_events(flags))

        async def _process_events(self, flags):
            while self._asyncio_loop._ready:
                await asyncio.sleep(0)
            super().processEvents(flags)
            await asyncio.sleep(0)

    class QAsyncioEventLoopPolicy(
        asyncio.DefaultEventLoopPolicy if sys.platform != "win32" else asyncio.WindowsProactorEventLoopPolicy
    ):
        def new_event_loop(self):
            qt_app = QCoreApplication.instance() or QCoreApplication(sys.argv)
            asyncio_loop = super().new_event_loop()
            return asyncio_loop

    def set_event_loop_policy(policy=None):
        """Set the asyncio event loop policy to use Qt integration."""
        if policy is None:
            policy = QAsyncioEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)

    def run(coro, *, debug=False):
        """Run the given coroutine with Qt event loop integration."""
        set_event_loop_policy()
        return asyncio.run(coro, debug=debug)

    # --------------------------------------------------------------------------- #
    # ThreadExecutor
    # --------------------------------------------------------------------------- #
    P = ParamSpec("P")
    R = TypeVar("R")

    class Task(QObject):
        """Wrapper for QRunnable tasks with Qt signal support."""
        finished = Signal(object, object)

        def __init__(self, fn: Callable, args: tuple, kwargs: dict, parent: Optional[QObject] = None):
            super().__init__(parent)
            self._fn = fn
            self._args = args
            self._kwargs = kwargs
            self._future: concurrent.futures.Future = concurrent.futures.Future()

        @property
        def future(self) -> concurrent.futures.Future:
            return self._future

        def run(self):
            if self._future.set_running_or_notify_cancel():
                try:
                    result = self._fn(*self._args, **self._kwargs)
                    self._future.set_result(result)
                    self.finished.emit(result, None)
                except Exception as e:
                    self._future.set_exception(e)
                    self.finished.emit(None, e)

    class FutureWatcher(QObject):
        """Watches a Future and emits signals on completion."""
        finished = Signal(object)
        error = Signal(object)

        def __init__(self, future: concurrent.futures.Future, parent: Optional[QObject] = None):
            super().__init__(parent)
            self._future = future
            future.add_done_callback(self._on_done)

        def _on_done(self, future: concurrent.futures.Future):
            try:
                result = future.result()
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(e)

    class ThreadPoolExecutorSignals(QObject):
        """Qt signals for ThreadExecutor events."""
        task_submitted = Signal(object)
        task_completed = Signal(object)
        task_failed = Signal(object, object)
        pool_shutdown = Signal()

    class ThreadExecutor(QObject, concurrent.futures.Executor):
        """A ThreadPoolExecutor that uses QThreadPool with Qt signal support."""

        def __init__(self, max_workers: Optional[int] = None, parent: Optional[QObject] = None):
            super().__init__(parent)
            self.signals = ThreadPoolExecutorSignals()
            self._pool = QThreadPool.globalInstance()
            self._max_workers = max_workers or self._pool.maxThreadCount()
            if max_workers:
                self._pool.setMaxThreadCount(max_workers)
            self._futures: List[concurrent.futures.Future] = []
            self._shutdown = False
            self._state_lock = threading.Lock()

        @property
        def max_workers(self) -> int:
            return self._max_workers

        def submit(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> concurrent.futures.Future[R]:
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
                task = Task(fn, args, kwargs)
                future = task.future
                self._futures.append(future)
                future.add_done_callback(self._on_future_done)
                try:
                    self.signals.task_submitted.emit(future)
                except RuntimeError:
                    pass

                class RunnableTask(QRunnable):
                    def __init__(self, t):
                        super().__init__()
                        self._task = t

                    def run(self):
                        self._task.run()

                self._pool.start(RunnableTask(task))
                return future

        def _on_future_done(self, future: concurrent.futures.Future):
            with self._state_lock:
                if future in self._futures:
                    self._futures.remove(future)
            try:
                exc = future.exception()
                if exc is not None:
                    self.signals.task_failed.emit(future, exc)
                else:
                    self.signals.task_completed.emit(future)
            except (concurrent.futures.CancelledError, RuntimeError):
                pass

        def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
            futures = [self.submit(fn, *args) for args in zip(*iterables)]
            start_time = time.monotonic() if timeout else None

            def result_iterator():
                for future in futures:
                    if timeout is not None:
                        elapsed = time.monotonic() - start_time
                        remaining = timeout - elapsed
                        if remaining <= 0:
                            raise TimeoutError()
                        yield future.result(timeout=remaining)
                    else:
                        yield future.result()

            return result_iterator()

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
            with self._state_lock:
                self._shutdown = True
            if cancel_futures:
                for future in self._futures:
                    future.cancel()
            if wait:
                self._pool.waitForDone()
            try:
                self.signals.pool_shutdown.emit()
            except RuntimeError:
                pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.shutdown(wait=True)
            return False

    # Aliases
    QThreadPoolExecutor = ThreadExecutor

    # --------------------------------------------------------------------------- #
    # ProcessPoolExecutor
    # --------------------------------------------------------------------------- #
    class ProcessPoolExecutorSignals(QObject):
        """Qt signals for ProcessPoolExecutor events."""
        task_submitted = Signal(object)
        task_completed = Signal(object)
        task_failed = Signal(object, object)
        pool_shutdown = Signal()

    class ProcessPoolExecutor(QObject, concurrent.futures.Executor):
        """ProcessPoolExecutor with Qt signal support."""

        def __init__(self, max_workers: Optional[int] = None, parent: Optional[QObject] = None):
            super().__init__(parent)
            self.signals = ProcessPoolExecutorSignals()
            self._max_workers = max_workers or multiprocessing.cpu_count()
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers)
            self._futures: List[concurrent.futures.Future] = []
            self._shutdown = False
            self._state_lock = threading.Lock()

        @property
        def max_workers(self) -> int:
            return self._max_workers

        def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
                future = self._executor.submit(fn, *args, **kwargs)
                self._futures.append(future)
                future.add_done_callback(self._on_future_done)
                try:
                    self.signals.task_submitted.emit(future)
                except RuntimeError:
                    pass
                return future

        def _on_future_done(self, future: concurrent.futures.Future):
            with self._state_lock:
                if future in self._futures:
                    self._futures.remove(future)
            try:
                exc = future.exception()
                if exc is not None:
                    self.signals.task_failed.emit(future, exc)
                else:
                    self.signals.task_completed.emit(future)
            except (concurrent.futures.CancelledError, RuntimeError):
                pass

        def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
            return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
            with self._state_lock:
                self._shutdown = True
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            try:
                self.signals.pool_shutdown.emit()
            except RuntimeError:
                pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.shutdown(wait=True)
            return False

    QProcessPoolExecutor = ProcessPoolExecutor

    # --------------------------------------------------------------------------- #
    # InterpreterPoolExecutor (process-based for embedded compatibility)
    # --------------------------------------------------------------------------- #
    INTERPRETER_POOL_AVAILABLE = True

    class InterpreterPoolExecutorSignals(QObject):
        """Qt signals for InterpreterPoolExecutor events."""
        task_submitted = Signal(object)
        task_completed = Signal(object)
        task_failed = Signal(object, object)
        pool_shutdown = Signal()

    class InterpreterPoolExecutor(QObject, concurrent.futures.Executor):
        """InterpreterPoolExecutor API using ProcessPoolExecutor backend for embedded contexts."""

        def __init__(self, max_workers: Optional[int] = None, parent: Optional[QObject] = None):
            super().__init__(parent)
            self.signals = InterpreterPoolExecutorSignals()
            self._max_workers = max_workers or multiprocessing.cpu_count()
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers)
            self._futures: List[concurrent.futures.Future] = []
            self._shutdown = False
            self._state_lock = threading.Lock()

        @property
        def max_workers(self) -> int:
            return self._max_workers

        def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
                future = self._executor.submit(fn, *args, **kwargs)
                self._futures.append(future)
                future.add_done_callback(self._on_future_done)
                try:
                    self.signals.task_submitted.emit(future)
                except RuntimeError:
                    pass
                return future

        def _on_future_done(self, future: concurrent.futures.Future):
            with self._state_lock:
                if future in self._futures:
                    self._futures.remove(future)
            try:
                exc = future.exception()
                if exc is not None:
                    self.signals.task_failed.emit(future, exc)
                else:
                    self.signals.task_completed.emit(future)
            except (concurrent.futures.CancelledError, RuntimeError):
                pass

        def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
            with self._state_lock:
                if self._shutdown:
                    raise RuntimeError("Cannot schedule new futures after shutdown.")
            return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
            with self._state_lock:
                self._shutdown = True
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            try:
                self.signals.pool_shutdown.emit()
            except RuntimeError:
                pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.shutdown(wait=True)
            return False

    QInterpreterPoolExecutor = InterpreterPoolExecutor

    QT_ASYNCIO_AVAILABLE = True

else:
    # Qt not available - define placeholders
    INTERPRETER_POOL_AVAILABLE = False
    QAsyncioEventLoop = None
    QAsyncioEventLoopPolicy = None
    set_event_loop_policy = None
    run = None
    Task = None
    FutureWatcher = None
    ThreadExecutor = None
    QThreadPoolExecutor = None
    ProcessPoolExecutor = None
    QProcessPoolExecutor = None
    InterpreterPoolExecutor = None
    QInterpreterPoolExecutor = None
    ThreadPoolExecutorSignals = None
    ProcessPoolExecutorSignals = None
    InterpreterPoolExecutorSignals = None


# =============================================================================
# WORKER MODULE
# =============================================================================

# Try to import QtAsyncio components if available
if QT_ASYNCIO_AVAILABLE:
    QTASYNCIO_ENABLED = True
    qt_set_event_loop_policy = set_event_loop_policy
else:
    QTASYNCIO_ENABLED = False
    qt_set_event_loop_policy = None


class ConnectionContext:
    """Context manager for a multiprocessing.connection.Connection."""

    def __init__(self, address: str, authkey: bytes | str, chunk_size: int = 1024):
        host, port_str = address.split(":")
        self.host = host
        self.port = int(port_str)
        if isinstance(authkey, str):
            authkey = bytes.fromhex(authkey)
        assert isinstance(authkey, bytes), f"Invalid authkey type: {type(authkey)}"
        self.authkey = authkey
        self._conn = None
        self.chunk_size = chunk_size

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    @property
    def conn(self) -> multiprocessing.connection.Connection:
        if self._conn is None:
            self._conn = multiprocessing.connection.Client(
                self.address, family="AF_INET", authkey=self.authkey
            )
            _logger.info(f"Connected to {self.address}")
        return self._conn

    def send_message(self, msg_type: str, data, **kwargs) -> bool:
        """Send a structured message through the connection."""
        try:
            if isinstance(data, list) and len(data) > self.chunk_size:
                message_id = uuid.uuid4().hex
                total_chunks = math.ceil(len(data) / self.chunk_size)
                for idx in range(total_chunks):
                    part = data[idx * self.chunk_size : (idx + 1) * self.chunk_size]
                    msg = {
                        "type": msg_type, "data": part, "timestamp": time.time(),
                        "message_id": message_id, "chunk_index": idx, "total_chunks": total_chunks,
                        **kwargs,
                    }
                    self.conn.send(msg)
                _logger.debug(f"Streamed {len(data)} items in {total_chunks} chunks under id {message_id}")
                return True

            msg = {"type": msg_type, "data": data, "timestamp": time.time(), **kwargs}
            self.conn.send(msg)
            _logger.debug(f"Sent single message: {msg_type}")
            return True
        except Exception as e:
            _logger.error(f"Failed to send message: {e}", exc_info=True)
            return False

    @property
    def closed(self):
        return self.conn.closed

    @property
    def readable(self):
        return self.conn.readable

    @property
    def writable(self):
        return self.conn.writable

    def fileno(self):
        return self.conn.fileno()

    def recv(self):
        return self.conn.recv()

    def poll(self, timeout=0.0):
        return self.conn.poll(timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            _logger.error("Connection closed by parent")
            return False
        if self.conn is not None:
            try:
                _logger.info("Closing worker-side connection.")
                self.conn.close()
            except Exception as e:
                _logger.error(f"Error closing connection: {e}")
        return True


class WorkerController:
    """Wrap an AsyncEventEmitter in its own event loop."""

    def __init__(self, emitter_instance: AsyncEventEmitter, use_qtasyncio: bool = False):
        self.emitter = emitter_instance
        self.use_qtasyncio = use_qtasyncio and QTASYNCIO_ENABLED

        if self.use_qtasyncio:
            _logger.info("Using QtAsyncio event loop for worker")
            qt_set_event_loop_policy()
            self.loop = asyncio.new_event_loop()
        else:
            self.loop = asyncio.new_event_loop()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._result = None
        self._started = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self._result = self.loop.run_until_complete(self.emitter.run())
        except Exception as e:
            _logger.error(f"Exception in worker thread loop: {e}", exc_info=True)
            self._result = None

    def start(self):
        if self._started:
            _logger.warning("Start called on an already started worker controller.")
            return
        self._thread.start()
        self._started = True

    def pause(self):
        if not self._started:
            _logger.warning("Pause called before worker controller was started.")
            return
        _logger.info("Pausing...")
        self.loop.call_soon_threadsafe(self.emitter.pause_evt.set)

    def resume(self):
        if not self._started:
            _logger.warning("Resume called before worker controller was started.")
            return
        _logger.info("Resuming...")
        self.loop.call_soon_threadsafe(self.emitter.pause_evt.clear)

    def stop(self):
        if not self._started:
            _logger.info("Stopping (controller not started)...")
            if hasattr(self.emitter, "stop_evt"):
                self.loop.call_soon_threadsafe(self.emitter.stop_evt.set)
            return
        _logger.info("Stopping...")
        self.loop.call_soon_threadsafe(self.emitter.stop_evt.set)

    def join(self):
        if not self._started:
            _logger.warning("Join called before worker controller was started.")
            return self._result
        if self._thread.is_alive():
            self._thread.join()
        else:
            _logger.info("Worker thread was not alive when join was called.")
        self._started = False
        return self._result

    def set_log_level(self, level):
        _logger.setLevel(level)
        if hasattr(self.emitter, "logger"):
            self.emitter.logger.setLevel(level)
        _logger.info(f"Log level set to {level} for controller and its emitter.")


class WorkerBase(WorkerProtocol):
    """Base class for worker implementations."""

    def __init__(
        self,
        async_emitter_class: typing.Type[AsyncEventEmitter] | None = None,
        emitter_args: dict | None = None,
        process_chunk_fn: typing.Callable | None = None,
    ):
        self.async_emitter_class = async_emitter_class
        self.emitter_args = emitter_args or {}
        self.process_chunk_fn = process_chunk_fn
        self.emitter_instance: AsyncEventEmitter | None = None
        self.controller: WorkerController | None = None
        self.conn: ConnectionContext | None = None

        self._commands = {
            "stop": self._handle_stop,
            "pause": self._handle_pause,
            "resume": self._handle_resume,
            "start": self._handle_start,
            "ping": self._handle_ping,
            "set_log_level": self._handle_set_log_level,
        }
        self.logger = get_logger(self.__class__.__name__)
        self._running = False

    def setup(self, **kwargs):
        """Default setup: Initializes the AsyncEventEmitter if provided."""
        if self.async_emitter_class:
            current_emitter_args = self.emitter_args.copy()
            if self.process_chunk_fn:
                current_emitter_args["process_chunk_fn"] = self.process_chunk_fn
            self.emitter_instance = self.async_emitter_class(**current_emitter_args)
            self._setup_default_event_handlers()
        else:
            self.logger.warning("No async_emitter_class provided for WorkerBase.")

    def _setup_default_event_handlers(self):
        if not self.emitter_instance:
            return

        @self.emitter_instance.on("run_started")
        def on_run_started():
            self.logger.info("Task starting")
            if self.conn:
                self.conn.send_message("progress", 0.0, status="running", stage="starting")

        @self.emitter_instance.on("run_finished")
        def on_run_finished(results):
            count = len(results) if hasattr(results, "__len__") else (1 if results is not None else 0)
            self.logger.info("Task finished, processed %s items/results.", count)
            if self.conn:
                self.conn.send_message("progress", 0.95, status="finalizing", stage="task_complete", items_count=count)

        @self.emitter_instance.on("stopped")
        def on_stopped():
            self.logger.info("Emitter shutting down")
            if self.conn:
                self.conn.send_message("status", "stopped", status="stopped")

        self.setup_custom_event_handlers()

    def setup_custom_event_handlers(self):
        """Subclasses can override this to add their own event handlers."""
        pass

    async def cleanup(self):
        if self.controller:
            self.controller.stop()
        if self.emitter_instance:
            await self.emitter_instance.shutdown()
        self.logger.info("Worker cleanup complete.")

    def handle_command(self, cmd: dict, conn: ConnectionContext) -> bool:
        cmd_type = cmd.get("command")
        if cmd_type is None:
            return True
        handler = self._commands.get(cmd_type)
        if handler:
            return handler(cmd, conn)
        return True

    def _handle_stop(self, cmd, conn):
        self.logger.info("Received stop command.")
        self._running = False
        if self.controller:
            self.controller.stop()
        conn.send_message("status", "stopped", status="stopped")
        return False

    def _handle_pause(self, cmd, conn):
        if self.controller:
            self.controller.pause()
            conn.send_message("status", "paused", status="paused")
        else:
            conn.send_message("error", "Not started", status="error")
        return True

    def _handle_resume(self, cmd, conn):
        if self.controller:
            self.controller.resume()
            conn.send_message("status", "resumed", status="running")
        else:
            conn.send_message("error", "Not started", status="error")
        return True

    def _handle_start(self, cmd, conn):
        if not self.emitter_instance:
            self.logger.error("Cannot start: emitter_instance not initialized.")
            conn.send_message("error", "Worker not properly configured (no emitter)", status="error")
            return True

        if self.controller and self.controller._started:
            self.logger.warning("Start command received, but already started.")
            conn.send_message("status", "already_running", status="running")
            return True

        self.logger.info("Received start command. Initializing and starting controller.")
        self.controller = WorkerController(self.emitter_instance)
        self.controller.start()
        conn.send_message("status", "started", status="running")
        return True

    def _handle_ping(self, cmd, conn):
        conn.send_message("status", "pong", status="running" if self._running else "idle")
        return True

    def _handle_set_log_level(self, cmd, conn):
        level = cmd.get("level")
        if level is None:
            self.logger.error("Log level is required for set_log_level command.")
            conn.send_message("error", "Log level not specified", status="error")
            return True

        try:
            if self.controller:
                self.controller.set_log_level(level)
            else:
                _logger.setLevel(level)
            self.logger.info(f"Log level set to {level}")
            conn.send_message("status", f"log_level_set:{level}", status="running")
        except Exception as e:
            self.logger.error(f"Failed to set log level: {e}", exc_info=True)
            conn.send_message("error", f"Failed to set log level: {str(e)}", status="error")
        return True

    def process(self, connection: ConnectionContext, **kwargs):
        """Main processing loop."""
        self.conn = connection
        self._running = True
        connection.send_message("status", "connected", status="ready")
        self.logger.info("Worker connected, awaiting commands...")

        try:
            while self._running:
                try:
                    if not connection.closed and not connection.poll(timeout=0.5):
                        if not self._running:
                            break
                        continue
                    if connection.closed:
                        self.logger.error("Connection closed by parent.")
                        self._running = False
                        break
                    cmd = connection.recv()
                    self.logger.debug(f"Received command: {cmd}")
                except EOFError:
                    self.logger.error("Connection closed by parent (EOFError).")
                    self._running = False
                    break
                except (ConnectionResetError, BrokenPipeError) as e:
                    self.logger.error(f"Connection error: {e}")
                    self._running = False
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error receiving command: {e}", exc_info=True)
                    continue

                if not self._running:
                    break

                if isinstance(cmd, dict):
                    if not self.handle_command(cmd, connection):
                        self._running = False
                        break
                elif cmd is None:
                    self.logger.info("Received None, likely connection closed.")
                    self._running = False
                    break
                else:
                    self.logger.warning(f"Received unexpected command type: {type(cmd)}")
                    connection.send_message("error", f"Expected dict command, got {type(cmd)}", status="error")

            if self.controller:
                self.logger.info("Process loop finished. Ensuring controller is stopped.")
                self.controller.stop()

        finally:
            self.logger.info("Process loop finalizing.")
            if self.controller:
                self.logger.info("Waiting for task controller to join...")
                results = self.controller.join()
                if results is not None:
                    self.send_results(connection, results)
                else:
                    self.logger.info("No results from controller.")
            else:
                self.logger.info("Controller was not active.")
            self.conn = None
            self.logger.info("Worker process loop ended.")

    def send_results(self, connection: ConnectionContext, results_data):
        """Formats and sends results."""
        if isinstance(results_data, IntervalSet):
            as_json = [{"address": s, "length": e - s, "end": e} for s, e in results_data.as_tuples()]
        elif isinstance(results_data, list):
            as_json = results_data
        elif results_data is not None:
            as_json = [results_data]
        else:
            as_json = []

        if as_json:
            self.logger.info(f"Sending {len(as_json)} results...")
            connection.send_message("status", "sending_results", status="sending_results")
            connection.send_message("result", as_json, status="success", count=len(as_json))
            connection.send_message("status", "results_sent", status="results_sent")
        else:
            self.logger.info("No results to send.")
            connection.send_message("result", [], status="success", count=0, note="No results generated")


# =============================================================================
# LAUNCHER MODULE (Qt-dependent)
# =============================================================================

if QT_AVAILABLE:
    class TemporarilyDisableNotifier:
        """Context manager to temporarily disable a QSocketNotifier."""
        def __init__(self, notifier):
            self.notifier = notifier
            self.was_enabled = False

        def __enter__(self):
            self.was_enabled = self.notifier.isEnabled()
            self.notifier.setEnabled(False)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.was_enabled:
                self.notifier.setEnabled(True)

    class ConnectionReader(QtCore.QThread):
        """Qt thread for reading messages from worker connection."""
        message_received = Signal(object)
        connection_closed = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.connection: multiprocessing.connection.Connection | None = None

        def set_connection(self, conn):
            self.connection = conn

        def run(self):
            if not self.connection:
                return
            try:
                while True:
                    msg = self.connection.recv()
                    self.message_received.emit(msg)
            except (EOFError, OSError):
                self.connection_closed.emit()
            except pickle.PickleError as e:
                _logger.error(f"Pickle error: {e}")
            finally:
                try:
                    self.connection.close()
                except Exception:
                    pass

    class QtListener(QtCore.QObject):
        """Qt-friendly wrapper for multiprocessing.connection.Listener."""
        family = "AF_INET"
        connection_accepted = Signal(object)
        connection_error = Signal(str)

        def __init__(self, address=None, backlog=1, authkey=None, parent=None):
            super().__init__(parent)
            self._listener = multiprocessing.connection.Listener(address, self.family, backlog, authkey)
            self._socket = self._listener._listener._socket
            self._notifier = QtCore.QSocketNotifier(self._socket.fileno(), QtCore.QSocketNotifier.Read, self)
            self._notifier.activated.connect(self._on_connection_ready)
            self._notifier.setEnabled(True)

        def _on_connection_ready(self):
            with TemporarilyDisableNotifier(self._notifier):
                try:
                    ready, _, _ = select.select([self._socket], [], [], 0)
                    if not ready:
                        return
                    conn = self.accept()
                    self.connection_accepted.emit(conn)
                except Exception as e:
                    _logger.error(f"Error accepting connection: {e}")
                    self.connection_error.emit(str(e))

        def accept(self):
            return self._listener.accept()

        def close(self):
            self._notifier.setEnabled(False)
            self._notifier.deleteLater()
            self._listener.close()

        @property
        def address(self):
            return self._listener.address

    class WorkerLauncher(QtCore.QProcess):
        """Manages external worker processes using QProcess."""
        processing_results = Signal(dict)
        error_occurred_msg = Signal(str)
        worker_message = Signal(object)
        worker_connected = Signal()
        worker_disconnected = Signal()

        def __init__(self, message_emitter: MessageEmitter | None = None, parent=None):
            super(WorkerLauncher, self).__init__(parent)
            self.message_emitter = message_emitter
            self.readyReadStandardOutput.connect(self._on_stdout)
            self.readyReadStandardError.connect(self._on_stderr)
            self.errorOccurred.connect(self._on_error)
            self.stateChanged.connect(self._on_state_changed)
            self.python_interpreter = MultiprocessingHelper.get_python_interpreter()
            self.listener = None
            self.connection = None
            self.authkey = None
            self.reader_thread = ConnectionReader(self)
            self.reader_thread.message_received.connect(self._on_worker_message)
            self.reader_thread.connection_closed.connect(self._on_connection_closed)
            self.connection_attempts = 0
            self.max_connection_attempts = 10
            self._streams: dict[str, dict] = {}

        def is_not_running(self):
            return self.state() == QtCore.QProcess.NotRunning

        def _on_worker_message(self, message):
            msg_id = message.get("message_id")
            if msg_id:
                idx = message["chunk_index"]
                total = message["total_chunks"]
                msg_type = message.get("type")
                stream = self._streams.setdefault(msg_id, {"type": msg_type, "chunks": {}, "total": total})
                stream["chunks"][idx] = message["data"]
                _logger.info("Received chunk %d/%d for %r (id=%s)", idx + 1, total, msg_type, msg_id)
                if len(stream["chunks"]) == total:
                    full = []
                    for i in range(total):
                        full.extend(stream["chunks"][i])
                    del self._streams[msg_id]
                    _logger.info("%r streaming complete (id=%s, %d items)", msg_type, msg_id, len(full))
                    self.processing_results.emit({"type": "result", "results": full, "status": "success"})
                return

            _logger.debug("Received message from worker: %r", message)
            self.worker_message.emit(message)

            if self.message_emitter and isinstance(message, dict):
                msg_type = message.get("type")
                if msg_type == "error":
                    self.message_emitter.emit_worker_error(message.get("error", "Unknown error"))
                elif msg_type == "result":
                    self.message_emitter.emit_worker_results({"results": message.get("data"), "status": message.get("status", "success")})
                elif msg_type == "status":
                    self.message_emitter.emit_worker_message(message)

        def _on_connection_closed(self):
            _logger.info("Worker IPC connection closed.")
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
            self.worker_disconnected.emit()
            if self.message_emitter:
                self.message_emitter.emit_worker_disconnected()

        def launch_worker(self, script_path: str, worker_args: dict):
            """Starts the worker script with specified arguments."""
            self._cleanup_resources()
            self.authkey = os.urandom(32)
            _logger.info(f"Generated Authkey: {self.authkey.hex()}")
            self.listener = QtListener(("localhost", 0), authkey=self.authkey, parent=self)
            address = self.listener.address
            _logger.info(f"Created listener on {address}")
            self.listener.connection_accepted.connect(self._on_connection_accepted)
            self.listener.connection_error.connect(self._on_connection_error)

            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHON_PATH", str(self.python_interpreter.parent))
            env.insert("PYTHON_BIN", str(self.python_interpreter.name))
            self.setProcessEnvironment(env)

            args = ["-u", script_path]
            args.extend(["--address", f"{address[0]}:{address[1]}", "--authkey", self.authkey.hex()])
            for key, value in worker_args.items():
                args.extend([f"--{key}", str(value)])

            _logger.info(f"Starting worker process: {self.python_interpreter} {args}")
            self.start(str(self.python_interpreter), args)

            if not self.waitForStarted(5000):
                _logger.error(f"Worker process failed to start: {self.errorString()}")
                self._cleanup_resources()
                return False

            self.connection_attempts = 0
            _logger.info("Worker process started. Beginning connection attempts...")
            return True

        def _on_connection_accepted(self, conn):
            _logger.info("Connection from worker accepted")
            self.reader_thread.set_connection(conn)
            if not self.reader_thread.isRunning():
                self.reader_thread.start()
            self.connection = conn
            self.worker_connected.emit()
            if self.message_emitter:
                self.message_emitter.emit_worker_connected()

        def _on_connection_error(self, error_msg):
            _logger.error(f"Connection error: {error_msg}")
            self.connection_attempts += 1
            if self.connection_attempts >= self.max_connection_attempts:
                self.error_occurred_msg.emit(f"Failed to connect to worker after {self.max_connection_attempts} attempts")
                self.stop_worker()

        def _cleanup_resources(self):
            if self.reader_thread.isRunning():
                self.reader_thread.terminate()
                self.reader_thread.wait()
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
            if self.listener:
                try:
                    self.listener.close()
                except:
                    pass
                self.listener = None

        def stop_worker(self):
            """Attempts to terminate the worker process gracefully."""
            if self.is_not_running() and not self.connection:
                _logger.debug("Worker process was already stopped.")
                return

            _logger.info("Attempting to stop worker process...")
            if self.connection:
                try:
                    _logger.info("Sending exit command...")
                    self.send_command({"command": "exit"})
                    if self.waitForFinished(1000):
                        _logger.info("Worker exited gracefully.")
                        self._cleanup_resources()
                        return
                except Exception as e:
                    _logger.error(f"Error sending exit command: {e}")

            if not self.is_not_running():
                _logger.warning("Worker did not exit gracefully, terminating...")
                self.terminate()
                if not self.waitForFinished(2000):
                    _logger.warning("Worker did not terminate, killing...")
                    self.kill()
                    self.waitForFinished(1000)

            self._cleanup_resources()
            _logger.info("Worker process shutdown complete.")

        def send_command(self, command):
            """Sends a command object via the connection."""
            if not self.connection:
                _logger.warning(f"Cannot send command '{command}', IPC connection not established.")
                return False
            _logger.debug(f"Sending command: {command}")
            try:
                self.connection.send(command)
                _logger.debug(f"Successfully sent command: {command}")
                return True
            except Exception as e:
                _logger.error(f"Failed to send command '{command}': {e}")
                self._on_connection_closed()
                return False

        def _on_stdout(self):
            out_bytes = self.readAllStandardOutput()
            if not out_bytes:
                return
            out = out_bytes.data().decode("utf-8", errors="replace")
            if out.strip():
                print(out.strip(), flush=True)

        def _on_stderr(self):
            data = self.readAllStandardError().data()
            if data:
                data = data.decode("utf-8", errors="replace").strip()
                print(data.strip(), file=sys.stderr, flush=True)

        def _on_error(self, error: QtCore.QProcess.ProcessError):
            error_map = {
                QtCore.QProcess.FailedToStart: "FailedToStart",
                QtCore.QProcess.Crashed: "Crashed",
                QtCore.QProcess.Timedout: "Timedout",
                QtCore.QProcess.ReadError: "ReadError",
                QtCore.QProcess.WriteError: "WriteError",
                QtCore.QProcess.UnknownError: "UnknownError",
            }
            error_str = error_map.get(error, f"UnknownError({error})")
            msg = f"Worker process error: {error_str} - {self.errorString()}"
            print(msg.strip(), file=sys.stderr, flush=True)
            self.error_occurred_msg.emit(msg)

        def _on_state_changed(self, state: QtCore.QProcess.ProcessState):
            state_map = {
                QtCore.QProcess.NotRunning: "NotRunning",
                QtCore.QProcess.Starting: "Starting",
                QtCore.QProcess.Running: "Running",
            }
            state_str = state_map.get(state, f"UnknownState({state})")
            _logger.info(f"Worker process state changed: {state_str}")

            if self.is_not_running():
                self._cleanup_resources()
                exit_code = self.exitCode()
                exit_status = self.exitStatus()
                exit_status_str = "NormalExit" if exit_status == QtCore.QProcess.NormalExit else "CrashExit"
                msg = f"Worker process exited with code {exit_code} ({exit_status_str})"
                _logger.info(msg)
                if exit_status == QtCore.QProcess.CrashExit:
                    self.error_occurred_msg.emit(msg)

else:
    # Qt not available - placeholders
    TemporarilyDisableNotifier = None
    ConnectionReader = None
    QtListener = None
    WorkerLauncher = None


# =============================================================================
# TASK RUNNER MODULE
# =============================================================================

if QT_AVAILABLE:
    class TaskRunner:
        """Simplified task runner with callback-based event handling."""

        def __init__(self, worker_script, worker_args, log_level=None, logger=None):
            if logger:
                self.logger = logger
            else:
                actual_log_level = log_level if log_level is not None else logging.INFO
                self.logger = get_logger(log_level=actual_log_level)
            self.message_emitter = MessageEmitter()
            self.launcher = WorkerLauncher(self.message_emitter)
            self.worker_script = worker_script
            self.worker_args = worker_args
            self._results_callback = None
            self._progress_callback = None

        def on_results(self, callback):
            """Register a callback for when worker results are received."""
            self._results_callback = callback
            self.message_emitter.on("worker_results", self._handle_results)

        def on_progress(self, callback):
            """Register a callback for progress updates."""
            self._progress_callback = callback
            self.message_emitter.on("worker_message", self._handle_progress)

        def start(self):
            """Start the worker task."""
            if self.launcher.launch_worker(self.worker_script, self.worker_args):
                self.logger.info("Worker launched successfully")
            else:
                self.logger.error("Failed to launch worker")

        def _handle_results(self, results):
            if self._results_callback:
                self._results_callback(results)

        def _handle_progress(self, message):
            if message.get("type") == "progress" and self._progress_callback:
                progress = message.get("progress", 0)
                status = message.get("status", "unknown")
                self._progress_callback(progress, status)

else:
    TaskRunner = None


# =============================================================================
# DATA PROCESSOR CORE (IDA-dependent)
# =============================================================================

class DataProcessorCore:
    """Core processor for managing deobfuscation tasks."""
    _shared_memory = None

    def __init__(self, message_emitter):
        if not isinstance(message_emitter, MessageEmitter):
            raise TypeError("message_emitter must be a MessageEmitter instance")
        self.message_emitter = message_emitter
        self.proc = None
        atexit.register(self.terminate)

    @staticmethod
    def get_section_data(section_name: str, max_size: int = 120 * 1024 * 1024, min_size: int = 1024) -> tuple[int, bytes]:
        """Get the data of a section by name."""
        try:
            import ida_bytes, ida_segment, idaapi
        except ImportError:
            _logger.error("IDA Pro modules not available")
            return 0, b""

        seg = ida_segment.get_segm_by_name(section_name)
        if not seg:
            _logger.error("Section %s not found", section_name)
            return idaapi.BADADDR, b""

        data_ea = seg.start_ea
        data_size = seg.end_ea - seg.start_ea

        if data_size > max_size:
            data_size = max_size
            _logger.warning("Limiting section data size to %s", humanize_bytes(data_size))
        elif data_size < min_size:
            _logger.error("%s section is too small (%s)", section_name, humanize_bytes(data_size))
            return idaapi.BADADDR, b""

        _logger.info("Reading %s from address %s", humanize_bytes(data_size), hex(data_ea))
        data_bytes = ida_bytes.get_bytes(data_ea, data_size)
        if not data_bytes or len(data_bytes) != data_size:
            _logger.error("Failed to read section data")
            return idaapi.BADADDR, b""

        return data_ea, data_bytes

    @staticmethod
    def from_range(start_ea: int, end_ea: int):
        """Get the data of a section by name and return the start address and the bytes."""
        try:
            import ida_bytes
        except ImportError:
            _logger.error("IDA Pro modules not available")
            return 0, b""
        data_bytes = ida_bytes.get_bytes(start_ea, end_ea - start_ea)
        return start_ea, data_bytes

    def run(self, start_ea: int, bytes_to_process: bytes, worker_script_path: str, **kwargs):
        """Run the deobfuscation process."""
        if not QT_AVAILABLE:
            _logger.error("Qt is required for DataProcessorCore.run()")
            return

        data_size = len(bytes_to_process)
        self._shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=data_size)
        self._shared_memory.buf[:data_size] = bytes_to_process

        self.proc = WorkerLauncher(self.message_emitter)
        worker_args = {
            "shm_name": self._shared_memory.name,
            "data_size": data_size,
            "start_ea": hex(start_ea),
            **kwargs,
        }

        try:
            import ida_ida
            worker_args["is64"] = "1" if ida_ida.inf_is_64bit() else "0"
        except ImportError:
            _logger.warning("IDA Pro modules not available, bitness detection skipped")

        if not self.proc.launch_worker(str(worker_script_path), worker_args):
            self.terminate()
            _logger.error("Failed to start worker process")
            return

    def terminate(self):
        """Terminate and clean up."""
        _logger.info("Terminating...")
        if self.proc and not self.proc.is_not_running():
            self.proc.stop_worker()
            self.proc = None
        self._cleanup_shared_memory()
        _logger.info("Terminated.")

    def _cleanup_shared_memory(self):
        if not self._shared_memory:
            return
        try:
            self._shared_memory.close()
            shm = multiprocessing.shared_memory.SharedMemory(self._shared_memory.name)
            shm.unlink()
            _logger.info("Shared memory unlinked: %s", self._shared_memory.name)
        except FileNotFoundError:
            _logger.warning("Shared memory already unlinked: %s", self._shared_memory.name)
        except PermissionError as e:
            _logger.error("Permission error unlinking shared memory: %s", e)
        except Exception as e:
            _logger.error("Unexpected error unlinking shared memory: %s", e, exc_info=True)
        finally:
            self._shared_memory = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Helpers
    "is_ida",
    "get_logger",
    "configure_logging",
    "MultiprocessingHelper",
    # Utils
    "humanize_bytes",
    "emit",
    "reify",
    "EventEmitter",
    "AsyncEventEmitter",
    "log_execution_time",
    "Range",
    "IntervalSet",
    "resolve_overlaps",
    "PatchManager",
    "DeferredPatchOp",
    "make_chunks",
    "shm_buffer",
    "execute_chunk_with_shm_view",
    "DataProcessorCore",
    # Protocols
    "WorkerProtocol",
    "MessageEmitter",
    # Qt compatibility
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
    # QtAsyncio
    "QAsyncioEventLoop",
    "QAsyncioEventLoopPolicy",
    "set_event_loop_policy",
    "run",
    "Task",
    "FutureWatcher",
    "ThreadExecutor",
    "QThreadPoolExecutor",
    "ThreadPoolExecutorSignals",
    "ProcessPoolExecutor",
    "QProcessPoolExecutor",
    "ProcessPoolExecutorSignals",
    "InterpreterPoolExecutor",
    "QInterpreterPoolExecutor",
    "InterpreterPoolExecutorSignals",
    "INTERPRETER_POOL_AVAILABLE",
    # Worker
    "ConnectionContext",
    "WorkerController",
    "WorkerBase",
    "QTASYNCIO_ENABLED",
    # Launcher
    "TemporarilyDisableNotifier",
    "ConnectionReader",
    "QtListener",
    "WorkerLauncher",
    # Task Runner
    "TaskRunner",
]
