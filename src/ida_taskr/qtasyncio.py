# QtAsyncio + ThreadExecutor + Worker utilities (single-file, typed, PySide6-compatible)
# Commit: qtproject/pyside-pyside-setup@072ffd057a29a694a0ad91894736bb4d0a88738e + extras
# Includes: asyncio integration, ThreadExecutor, Task/FutureWatcher, thread_worker utils

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import inspect
import logging
import sys
import threading
import time
import warnings
import weakref
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Generator,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

from .qt_compat import QtCore, Signal, Slot, QT_AVAILABLE

if not QT_AVAILABLE:
    raise ImportError(
        "QtAsyncio module requires Qt (PyQt5 or PySide6) to be available. "
        "Please install PyQt5 or PySide6."
    )

# Import Qt components we need
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

# Try to import Q_ARG and QRunnable with compatibility handling
try:
    from PySide6.QtCore import Q_ARG, QRunnable
except ImportError:
    try:
        from PyQt5.QtCore import Q_ARG, QRunnable
    except ImportError:
        # Fallback if neither is available
        Q_ARG = None
        QRunnable = object  # type: ignore

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any

# Type variable for Future
Future = concurrent.futures.Future

# --------------------------------------------------------------------------- #
# Asyncio integration (original QtAsyncio module)
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
    asyncio.DefaultEventLoopPolicy
    if sys.platform != "win32"
    else asyncio.WindowsProactorEventLoopPolicy
):
    def new_event_loop(self):
        qt_app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        asyncio_loop = super().new_event_loop()
        qt_loop = QAsyncioEventLoop(asyncio_loop, qt_app)

        def _wakeup_qt():
            qt_loop.wakeUp()

        asyncio_loop.call_soon(_wakeup_qt)
        timer = QTimer(qt_app)
        timer.timeout.connect(lambda: asyncio_loop.run_until_complete(asyncio.sleep(0)))
        timer.start(10)
        return asyncio_loop


def run(coroutine_or_future):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine_or_future)


def set_event_loop_policy():
    """Set the Qt-compatible asyncio event loop policy."""
    if not isinstance(asyncio.get_event_loop_policy(), QAsyncioEventLoopPolicy):
        asyncio.set_event_loop_policy(QAsyncioEventLoopPolicy())


# --------------------------------------------------------------------------- #
# ThreadExecutor + Task + FutureWatcher + utilities
# --------------------------------------------------------------------------- #
@contextmanager
def locked(mutex):
    mutex.lock()
    try:
        yield
    finally:
        mutex.unlock()


class _TaskDepotThread(QThread):
    _lock = threading.Lock()
    _instance: Optional["_TaskDepotThread"] = None

    def __new__(cls):
        if _TaskDepotThread._instance is not None:
            raise RuntimeError("Already exists")
        return super().__new__(cls)

    def __init__(self):
        super().__init__()
        self.start()
        self.moveToThread(self)
        atexit.register(self._cleanup)

    def _cleanup(self):
        self.quit()
        self.wait()

    @staticmethod
    def instance() -> "_TaskDepotThread":
        with _TaskDepotThread._lock:
            if _TaskDepotThread._instance is None:
                _TaskDepotThread._instance = _TaskDepotThread()
            return _TaskDepotThread._instance

    @Slot(object, object)
    def transfer(self, obj: QObject, thread: QThread):
        assert obj.thread() is self
        assert QThread.currentThread() is self
        obj.moveToThread(thread)


class _TaskRunnable(QRunnable):
    def __init__(self, future, task, args, kwargs):
        super().__init__()
        self.future = future
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.eventLoop: Optional[QEventLoop] = None

    def run(self):
        self.eventLoop = QEventLoop()
        self.eventLoop.processEvents()

        if Q_ARG is not None:
            QMetaObject.invokeMethod(
                self.task.thread(),
                "transfer",
                Qt.BlockingQueuedConnection,
                Q_ARG(object, self.task),
                Q_ARG(object, QThread.currentThread()),
            )

        self.eventLoop.processEvents()
        self.task.start()
        self.task.finished.connect(self.eventLoop.quit)
        self.task.cancelled.connect(self.eventLoop.quit)
        self.eventLoop.exec_()


class FutureRunnable(QRunnable):
    def __init__(self, future, func, args, kwargs):
        super().__init__()
        self.future = future
        self.task = (func, args, kwargs)

    def run(self):
        try:
            if not self.future.set_running_or_notify_cancel():
                return
            func, args, kwargs = self.task
            result = func(*args, **kwargs)
            self.future.set_result(result)
        except BaseException as ex:
            self.future.set_exception(ex)
        except BaseException:
            logging.getLogger(__name__).critical("Exception in worker thread.", exc_info=True)


class ThreadPoolExecutorSignals(QObject):
    """Qt signals for ThreadExecutor events."""
    task_submitted = Signal(object)  # future
    task_completed = Signal(object)  # future
    task_failed = Signal(object, object)  # future, exception
    pool_shutdown = Signal()


class ThreadExecutor(QObject, concurrent.futures.Executor):
    """
    A ThreadExecutor that provides concurrent.futures.ThreadPoolExecutor-like API
    using Qt's QThreadPool with optional Qt signal integration.

    Usage:
        executor = ThreadExecutor()

        # Standard concurrent.futures API
        future = executor.submit(task, arg1, arg2)
        result = future.result()

        # Or with Qt signals
        executor.signals.task_completed.connect(on_task_done)
        future = executor.submit(task, arg1, arg2)

        executor.shutdown(wait=True)
    """

    def __init__(
        self,
        parent: Optional[QObject] = None,
        threadPool: Optional[QThreadPool] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__(parent)
        self.signals = ThreadPoolExecutorSignals()
        self._threadPool = threadPool or QThreadPool.globalInstance()

        # Set max thread count if specified
        if max_workers is not None:
            self._threadPool.setMaxThreadCount(max_workers)

        self._max_workers = max_workers or self._threadPool.maxThreadCount()
        self._depot_thread: Optional[_TaskDepotThread] = None
        self._futures: List[Future] = []
        self._shutdown = False
        self._state_lock = threading.Lock()

    @property
    def max_workers(self) -> int:
        """Return the maximum number of worker threads."""
        return self._max_workers

    def _get_depot_thread(self) -> _TaskDepotThread:
        if self._depot_thread is None:
            self._depot_thread = _TaskDepotThread.instance()
        return self._depot_thread

    def submit(self, func, *args, **kwargs) -> Future:
        """
        Submit a callable to be executed in a worker thread.

        Args:
            func: A callable to execute
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            A Future representing the pending execution

        Raises:
            RuntimeError: If the executor has been shut down
        """
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown.")
            if isinstance(func, Task):
                warnings.warn("Use `submit_task` to run `Task`s", DeprecationWarning, stacklevel=2)
                f, runnable = self.__make_task_runnable(func)
            else:
                f = Future()
                runnable = FutureRunnable(f, func, args, kwargs)
            self._futures.append(f)
            f.add_done_callback(self._future_done)
            self._threadPool.start(runnable)

            # Emit task_submitted signal
            try:
                self.signals.task_submitted.emit(f)
            except RuntimeError:
                pass  # Qt object may have been deleted

            return f

    def __make_task_runnable(self, task: Task) -> Tuple[Future, _TaskRunnable]:
        if task.thread() is not QThread.currentThread():
            raise ValueError("Can only submit Tasks from its own thread.")
        if task.parent() is not None:
            raise ValueError("Cannot submit Tasks with a parent.")
        task.moveToThread(self._get_depot_thread())
        f = task.future()
        runnable = _TaskRunnable(f, task, (), {})
        return f, runnable

    def map(
        self,
        fn: Callable,
        *iterables,
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ):
        """
        Map a function over iterables, executing in parallel threads.

        Args:
            fn: A callable
            *iterables: Iterables of arguments
            timeout: Maximum time to wait for results
            chunksize: Ignored (for API compatibility with ProcessPoolExecutor)

        Returns:
            Iterator of results in the same order as the input iterables
        """
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown.")

        # Submit all tasks
        futures = [self.submit(fn, *args) for args in zip(*iterables)]

        # Yield results in order
        for future in futures:
            yield future.result(timeout=timeout)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """
        Shutdown the executor.

        Args:
            wait: If True, wait for all pending futures to complete
            cancel_futures: If True, cancel all pending futures (best effort)
        """
        with self._state_lock:
            self._shutdown = True
            futures = list(self._futures)

        if cancel_futures:
            for f in futures:
                f.cancel()

        if wait:
            concurrent.futures.wait(futures)

        try:
            self.signals.pool_shutdown.emit()
        except RuntimeError:
            pass  # Qt object may have been deleted

    def _future_done(self, future: Future):
        """Called when a future completes - emits appropriate Qt signals."""
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
            pass  # Future was cancelled or Qt object deleted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class Task(QObject):
    started = Signal()
    finished = Signal()
    cancelled = Signal()
    resultReady = Signal(object)
    exceptionReady = Signal(Exception)

    __ExecuteCall = QEvent.registerEventType()

    def __init__(self, parent: Optional[QObject] = None, function: Optional[Callable] = None):
        super().__init__(parent)
        warnings.warn("`Task` has been deprecated", PendingDeprecationWarning, stacklevel=2)
        self.function = function
        self._future: Future = Future()

    def run(self):
        if self.function is None:
            raise NotImplementedError
        return self.function()

    def start(self):
        QCoreApplication.postEvent(self, QEvent(Task.__ExecuteCall))

    def future(self) -> Future:
        return self._future

    def result(self, timeout: Optional[float] = None):
        return self._future.result(timeout)

    def _execute(self):
        try:
            if not self._future.set_running_or_notify_cancel():
                self.cancelled.emit()
                return
            self.started.emit()
            result = self.run()
            self._future.set_result(result)
            self.resultReady.emit(result)
        except BaseException as ex:
            self._future.set_exception(ex)
            self.exceptionReady.emit(ex)
        finally:
            self.finished.emit()

    def customEvent(self, event: QEvent):
        if event.type() == Task.__ExecuteCall:
            self._execute()
        else:
            super().customEvent(event)


class FutureWatcher(QObject):
    done = Signal(Future)
    finished = Signal(Future)
    cancelled = Signal(Future)
    resultReady = Signal(object)
    exceptionReady = Signal(BaseException)

    __FutureDone = QEvent.registerEventType()

    def __init__(self, future: Optional[Future] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.__future: Optional[Future] = None
        if future is not None:
            self.setFuture(future)

    def setFuture(self, future: Future):
        if self.__future is not None:
            raise RuntimeError("Future already set")
        self.__future = future

        def on_done(f):
            if (selfref := weakref.ref(self)) is None:
                return
            try:
                QCoreApplication.postEvent(selfref(), QEvent(FutureWatcher.__FutureDone))
            except RuntimeError:
                pass

        future.add_done_callback(on_done)

    def future(self) -> Future:
        return self.__future

    def __emitSignals(self):
        f = self.__future
        if f.cancelled():
            self.cancelled.emit(f)
            self.done.emit(f)
        else:
            self.finished.emit(f)
            self.done.emit(f)
            if exc := f.exception():
                self.exceptionReady.emit(exc)
            else:
                self.resultReady.emit(f.result())

    def customEvent(self, event: QEvent):
        if event.type() == FutureWatcher.__FutureDone:
            self.__emitSignals()
        super().customEvent(event)


# --------------------------------------------------------------------------- #
# Modern thread_worker utilities (QRunnable-based)
# --------------------------------------------------------------------------- #
_T = TypeVar("_T")
_R = TypeVar("_R")
_Y = TypeVar("_Y")
_S = TypeVar("_S")
_P = ParamSpec("_P")


class WorkerBaseSignals(QObject):
    started = Signal()
    finished = Signal()
    _finished = Signal(object)
    returned = Signal(object)
    errored = Signal(object)
    warned = Signal(tuple)


class WorkerBase(QRunnable, Generic[_R]):
    _worker_set: set[Self] = set()

    def __init__(self, SignalsClass=WorkerBaseSignals):
        super().__init__()
        self._abort_requested = False
        self._running = False
        self.signals = SignalsClass()

    def __getattr__(self, name: str):
        attr = getattr(self.signals.__class__, name, None)
        if isinstance(attr, Signal):
            return getattr(self.signals, name)
        raise AttributeError(f"{self.__class__.__name__!r} has no attribute {name!r}")

    def quit(self):
        self._abort_requested = True

    @property
    def abort_requested(self) -> bool:
        return self._abort_requested

    @property
    def is_running(self) -> bool:
        return self._running

    def run(self):
        self.started.emit()
        self._running = True
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("always")
                warnings.showwarning = lambda *w: self.warned.emit(w)
                result = self.work()
            if isinstance(result, Exception):
                raise result
            if not self.abort_requested:
                self.returned.emit(result)
        except Exception as e:
            self.errored.emit(e)
        finally:
            self._running = False
            self.finished.emit()
            self._finished.emit(self)

    def work(self) -> _R | Exception:
        raise NotImplementedError

    def start(self):
        if self in self._worker_set:
            raise RuntimeError("Worker already started")
        self._worker_set.add(self)
        self._finished.connect(lambda w: self._worker_set.discard(w))
        QThreadPool.globalInstance().start(self)


class FunctionWorker(WorkerBase[_R]):
    def __init__(self, func: Callable[_P, _R], *args, **kwargs):
        if inspect.isgeneratorfunction(func):
            raise TypeError("Use GeneratorWorker for generator functions")
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def work(self) -> _R:
        return self._func(*self._args, **self._kwargs)


class GeneratorWorkerSignals(WorkerBaseSignals):
    yielded = Signal(object)
    paused = Signal()
    resumed = Signal()
    aborted = Signal()


class GeneratorWorker(WorkerBase, Generic[_Y, _S, _R]):
    yielded: Signal
    paused: Signal
    resumed: Signal
    aborted: Signal

    def __init__(self, func: Callable[..., Generator[_Y, Optional[_S], _R]], *args, **kwargs):
        if not inspect.isgeneratorfunction(func):
            raise TypeError("Use FunctionWorker for regular functions")
        super().__init__(SignalsClass=GeneratorWorkerSignals)
        self._gen = func(*args, **kwargs)
        self._incoming: Optional[_S] = None
        self._pause_requested = self._resume_requested = self._paused = False
        self._pause_interval = 0.01

    def work(self) -> _R | None:
        while True:
            if self.abort_requested:
                self.aborted.emit()
                break
            if self._paused:
                if self._resume_requested:
                    self._paused = self._resume_requested = False
                    self.resumed.emit()
                else:
                    time.sleep(self._pause_interval)
                    continue
            elif self._pause_requested:
                self._paused = True
                self._pause_requested = False
                self.paused.emit()
                continue
            try:
                output = self._gen.send(self._incoming)
                self.yielded.emit(output)
                self._incoming = None
            except StopIteration as e:
                return e.value
        return None

    def send(self, value: _S):
        self._incoming = value

    def toggle_pause(self):
        if self._paused:
            self._resume_requested = True
        else:
            self._pause_requested = True


@overload
def create_worker(
    func: Callable[_P, Generator[_Y, _S, _R]], *args, **kwargs
) -> GeneratorWorker[_Y, _S, _R]: ...
@overload
def create_worker(func: Callable[_P, _R], *args, **kwargs) -> FunctionWorker[_R]: ...

def create_worker(func, *args, **kwargs):
    WorkerCls = GeneratorWorker if inspect.isgeneratorfunction(func) else FunctionWorker
    return WorkerCls(func, *args, **kwargs)


def thread_worker(func=None, **dec_kwargs):
    def decorator(f):
        @wraps(f)
        def wrapper(*a, **kw):
            return create_worker(f, *a, **kw, **dec_kwargs)
        return wrapper
    return decorator if func is None else decorator(func)


# --------------------------------------------------------------------------- #
# QThread-based worker helper (alternative to QRunnable)
# --------------------------------------------------------------------------- #
def new_worker_qthread(
    Worker: type[QObject],
    *args,
    _start_thread: bool = False,
    _connect: Optional[Dict[str, Callable]] = None,
    **kwargs,
):
    thread = QThread()
    worker = Worker(*args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.work)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    if _connect:
        for sig, slot in _connect.items():
            getattr(worker, sig).connect(slot)
    if _start_thread:
        thread.start()
    return worker, thread

# --------------------------------------------------------------------------- #
# ProcessPoolExecutor - multiprocessing-based executor with Qt signal support
# --------------------------------------------------------------------------- #
import multiprocessing
import multiprocessing.pool
import queue


class ProcessPoolExecutorSignals(QObject):
    """Qt signals for ProcessPoolExecutor events."""
    task_submitted = Signal(object)  # future
    task_completed = Signal(object)  # future
    task_failed = Signal(object, object)  # future, exception
    pool_shutdown = Signal()


class ProcessPoolExecutor(QObject, concurrent.futures.Executor):
    """
    A ProcessPoolExecutor that provides concurrent.futures.ProcessPoolExecutor API
    with optional Qt signal integration for task completion notifications.

    Unlike ThreadExecutor which uses QThreadPool, this uses Python's multiprocessing
    for true parallel execution of CPU-bound tasks across multiple cores.

    Usage:
        executor = ProcessPoolExecutor(max_workers=4)

        # Standard concurrent.futures API
        future = executor.submit(cpu_bound_task, arg1, arg2)
        result = future.result()

        # Or with Qt signals
        executor.signals.task_completed.connect(on_task_done)
        future = executor.submit(cpu_bound_task, arg1, arg2)

        executor.shutdown(wait=True)

    Note: Functions submitted must be picklable (module-level functions, not lambdas
    or closures that capture unpicklable objects).
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
        initializer: Optional[Callable[..., None]] = None,
        initargs: Tuple = (),
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.signals = ProcessPoolExecutorSignals()
        self._max_workers = max_workers or multiprocessing.cpu_count()
        self._mp_context = mp_context
        self._initializer = initializer
        self._initargs = initargs

        # Create the underlying ProcessPoolExecutor
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=self._mp_context,
            initializer=self._initializer,
            initargs=self._initargs,
        )

        self._futures: List[concurrent.futures.Future] = []
        self._shutdown = False
        self._state_lock = threading.Lock()

    @property
    def max_workers(self) -> int:
        """Return the maximum number of worker processes."""
        return self._max_workers

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """
        Submit a callable to be executed in a worker process.

        Args:
            fn: A picklable callable to execute
            *args: Positional arguments for the callable
            **kwargs: Keyword arguments for the callable

        Returns:
            A Future representing the pending execution

        Raises:
            RuntimeError: If the executor has been shut down
        """
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown.")

            future = self._executor.submit(fn, *args, **kwargs)
            self._futures.append(future)

            # Add callback for Qt signal emission
            future.add_done_callback(self._on_future_done)

            # Emit task_submitted signal
            try:
                self.signals.task_submitted.emit(future)
            except RuntimeError:
                pass  # Qt object may have been deleted

            return future

    def _on_future_done(self, future: concurrent.futures.Future):
        """Called when a future completes - emits appropriate Qt signals."""
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
            pass  # Future was cancelled or Qt object deleted

    def map(
        self,
        fn: Callable,
        *iterables,
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ):
        """
        Map a function over iterables, executing in parallel.

        Args:
            fn: A picklable callable
            *iterables: Iterables of arguments
            timeout: Maximum time to wait for results
            chunksize: Size of chunks for efficiency (larger = fewer IPC calls)

        Returns:
            Iterator of results in the same order as the input iterables

        Raises:
            RuntimeError: If the executor has been shut down
        """
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown.")

        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """
        Shutdown the executor.

        Args:
            wait: If True, wait for all pending futures to complete
            cancel_futures: If True, cancel all pending futures
        """
        with self._state_lock:
            self._shutdown = True

        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

        try:
            self.signals.pool_shutdown.emit()
        except RuntimeError:
            pass  # Qt object may have been deleted

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


# Alias for consistency with naming convention
QProcessPoolExecutor = ProcessPoolExecutor
QThreadPoolExecutor = ThreadExecutor  # Alias


# Single-file ready-to-use module.
# No external dependencies beyond PyQt5/PySide6.

__all__ = [
    # Asyncio integration
    "QAsyncioEventLoop",
    "QAsyncioEventLoopPolicy",
    "run",
    "set_event_loop_policy",
    # Thread executor
    "ThreadExecutor",
    "QThreadPoolExecutor",  # Alias
    "ThreadPoolExecutorSignals",
    "Task",
    "FutureWatcher",
    "Future",
    # Process executor
    "ProcessPoolExecutor",
    "QProcessPoolExecutor",  # Alias
    "ProcessPoolExecutorSignals",
    # Worker utilities
    "WorkerBase",
    "WorkerBaseSignals",
    "FunctionWorker",
    "GeneratorWorker",
    "GeneratorWorkerSignals",
    "create_worker",
    "thread_worker",
    "new_worker_qthread",
]
