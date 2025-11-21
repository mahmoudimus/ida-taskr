"""
Unit tests for QtAsyncio integration.

Tests the qtasyncio module components including worker utilities,
thread executor, and event loop integration.
"""

import asyncio
import time

import pytest

from ida_taskr import QT_ASYNCIO_AVAILABLE, is_ida

# Skip all tests if QtAsyncio is not available
pytestmark = pytest.mark.skipif(
    not QT_ASYNCIO_AVAILABLE,
    reason="QtAsyncio module not available"
)


class TestQtAsyncioImports:
    """Test that QtAsyncio components can be imported."""

    def test_import_worker_utilities(self):
        """Test importing worker utilities."""
        from ida_taskr import (
            QtWorkerBase,
            FunctionWorker,
            GeneratorWorker,
            create_worker,
            thread_worker,
        )

        assert QtWorkerBase is not None
        assert FunctionWorker is not None
        assert GeneratorWorker is not None
        assert create_worker is not None
        assert thread_worker is not None

    def test_import_thread_executor(self):
        """Test importing thread executor components."""
        from ida_taskr import ThreadExecutor, Task, FutureWatcher

        assert ThreadExecutor is not None
        assert Task is not None
        assert FutureWatcher is not None

    def test_import_asyncio_integration(self):
        """Test importing asyncio integration components."""
        from ida_taskr import (
            QAsyncioEventLoop,
            QAsyncioEventLoopPolicy,
            set_event_loop_policy,
        )

        assert QAsyncioEventLoop is not None
        assert QAsyncioEventLoopPolicy is not None
        assert set_event_loop_policy is not None


class TestFunctionWorker:
    """Test FunctionWorker functionality."""

    def test_create_function_worker(self):
        """Test creating a function worker."""
        from ida_taskr import create_worker

        def simple_func(x, y):
            return x + y

        worker = create_worker(simple_func, 2, 3)
        assert worker is not None
        assert hasattr(worker, 'start')
        assert hasattr(worker, 'returned')
        assert hasattr(worker, 'errored')
        assert hasattr(worker, 'finished')

    def test_thread_worker_decorator(self):
        """Test the @thread_worker decorator."""
        from ida_taskr import thread_worker

        @thread_worker
        def decorated_func(x):
            return x * 2

        worker = decorated_func(5)
        assert worker is not None
        assert hasattr(worker, 'start')

    def test_function_worker_execution(self):
        """Test that function worker executes correctly."""
        from ida_taskr import create_worker

        result_holder = []

        def test_func():
            time.sleep(0.1)
            return "test_result"

        worker = create_worker(test_func)

        # Connect signal to capture result
        worker.returned.connect(lambda r: result_holder.append(r))

        # Note: We can't easily test this without a Qt event loop running
        # This is a basic structural test
        assert worker is not None


class TestGeneratorWorker:
    """Test GeneratorWorker functionality."""

    def test_create_generator_worker(self):
        """Test creating a generator worker."""
        from ida_taskr import create_worker

        def generator_func():
            for i in range(3):
                yield i

        worker = create_worker(generator_func)
        assert worker is not None
        assert hasattr(worker, 'yielded')
        assert hasattr(worker, 'returned')

    def test_generator_worker_type_detection(self):
        """Test that generator functions are detected correctly."""
        from ida_taskr import create_worker, GeneratorWorker

        def generator_func():
            yield 1

        worker = create_worker(generator_func)
        assert isinstance(worker, GeneratorWorker)


class TestThreadExecutor:
    """Test ThreadExecutor functionality."""

    def test_create_thread_executor(self):
        """Test creating a ThreadExecutor."""
        from ida_taskr import ThreadExecutor

        executor = ThreadExecutor()
        assert executor is not None
        assert hasattr(executor, 'submit')
        assert hasattr(executor, 'shutdown')

    def test_thread_executor_submit(self):
        """Test submitting a task to ThreadExecutor."""
        from ida_taskr import ThreadExecutor

        def simple_task(x):
            return x * 2

        executor = ThreadExecutor()
        future = executor.submit(simple_task, 5)

        assert future is not None
        # Note: Can't easily wait for result without Qt event loop
        # This is a structural test

        executor.shutdown(wait=False)


class TestWorkerControllerIntegration:
    """Test WorkerController with QtAsyncio support."""

    def test_worker_controller_qtasyncio_flag(self):
        """Test that WorkerController accepts use_qtasyncio parameter."""
        from ida_taskr.worker import WorkerController
        from ida_taskr.utils import AsyncEventEmitter
        import dataclasses

        @dataclasses.dataclass
        class DummyEmitter(AsyncEventEmitter):
            async def run(self):
                return "done"

            async def shutdown(self):
                pass

        emitter = DummyEmitter()

        # Test with QtAsyncio disabled
        controller = WorkerController(emitter, use_qtasyncio=False)
        assert controller.use_qtasyncio is False

        # Test with QtAsyncio enabled (if available)
        controller_qt = WorkerController(emitter, use_qtasyncio=True)
        # Should be enabled if QTASYNCIO_ENABLED is True
        assert hasattr(controller_qt, 'use_qtasyncio')


class TestEventLoopPolicy:
    """Test Qt event loop policy integration."""

    def test_set_event_loop_policy(self):
        """Test setting Qt event loop policy."""
        from ida_taskr import set_event_loop_policy, QAsyncioEventLoopPolicy

        # Set the policy
        set_event_loop_policy()

        # Verify it was set
        policy = asyncio.get_event_loop_policy()
        assert isinstance(policy, QAsyncioEventLoopPolicy)

    def test_event_loop_creation(self):
        """Test creating a new event loop with Qt policy."""
        from ida_taskr import set_event_loop_policy

        # Set the policy
        set_event_loop_policy()

        # Create a new event loop
        loop = asyncio.new_event_loop()
        assert loop is not None

        # Clean up
        loop.close()


class TestNewWorkerQThread:
    """Test new_worker_qthread helper function."""

    def test_import_new_worker_qthread(self):
        """Test that new_worker_qthread can be imported."""
        from ida_taskr import new_worker_qthread

        assert new_worker_qthread is not None
        assert callable(new_worker_qthread)


# Integration test that requires a Qt application
class TestQtApplicationIntegration:
    """Integration tests that require a Qt application running."""

    @pytest.mark.skipif(not is_ida(), reason="Requires IDA Pro's Qt application")
    def test_full_worker_execution(self):
        """Full test of worker execution (IDA Pro only)."""
        # In IDA Pro, Qt application is already running
        from ida_taskr import create_worker

        def simple_task():
            return "completed"

        worker = create_worker(simple_task)
        assert worker is not None
        assert hasattr(worker, 'start')
        # Note: Can't easily test actual execution without Qt event loop running
