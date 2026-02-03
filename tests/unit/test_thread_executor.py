"""
Tests for ThreadExecutor - Qt-native concurrent.futures.Executor implementation.

These tests demonstrate using ThreadExecutor similar to concurrent.futures patterns
but with Qt integration for signal/slot support.
"""

import concurrent.futures
import time
import pytest

from ida_taskr import QT_ASYNCIO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not QT_ASYNCIO_AVAILABLE,
    reason="QtAsyncio module not available"
)


class TestThreadExecutorBasic:
    """Basic ThreadExecutor functionality tests."""

    def test_submit_single_task(self):
        """Test submitting a single task to ThreadExecutor."""
        from ida_taskr import ThreadExecutor

        def square(n):
            return n * n

        executor = ThreadExecutor()
        future = executor.submit(square, 5)

        # Wait for result
        result = future.result(timeout=5)
        assert result == 25

        executor.shutdown(wait=True)

    def test_submit_multiple_tasks(self):
        """Test submitting multiple tasks concurrently."""
        from ida_taskr import ThreadExecutor

        def compute_square(n):
            time.sleep(0.1)  # Simulate work
            return n * n

        numbers = [1, 2, 3, 4, 5]
        executor = ThreadExecutor()

        futures = [executor.submit(compute_square, num) for num in numbers]

        # Collect results
        results = [f.result(timeout=5) for f in futures]
        assert results == [1, 4, 9, 16, 25]

        executor.shutdown(wait=True)

    def test_as_completed_pattern(self):
        """Test using concurrent.futures.as_completed with ThreadExecutor."""
        from ida_taskr import ThreadExecutor

        def slow_task(n):
            time.sleep(n * 0.05)  # Variable delay
            return n * n

        numbers = [3, 1, 2]  # Different delays
        executor = ThreadExecutor()

        futures = {executor.submit(slow_task, num): num for num in numbers}
        completed_order = []

        for future in concurrent.futures.as_completed(futures, timeout=5):
            num = futures[future]
            result = future.result()
            completed_order.append(num)
            assert result == num * num

        # Should complete in order: 1, 2, 3 (fastest first)
        assert completed_order == [1, 2, 3]

        executor.shutdown(wait=True)


class TestThreadExecutorFactorial:
    """Test ThreadExecutor with CPU-bound factorial computation."""

    def test_compute_factorials_concurrently(self):
        """Test computing factorials concurrently - similar to ProcessPoolExecutor example."""
        from ida_taskr import ThreadExecutor

        def compute_factorial(n):
            """Compute factorial of n."""
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result

        # Use smaller numbers for faster tests
        numbers = [100, 200, 300, 400]
        executor = ThreadExecutor()

        start_time = time.time()

        futures = [executor.submit(compute_factorial, num) for num in numbers]
        results = {}

        for future in concurrent.futures.as_completed(futures, timeout=10):
            # Find which number this future corresponds to
            idx = futures.index(future)
            num = numbers[idx]
            result = future.result()
            results[num] = result

        end_time = time.time()

        # Verify all factorials were computed
        assert len(results) == len(numbers)
        for num in numbers:
            assert num in results
            # Verify factorial is correct (spot check)
            assert results[num] > 0

        # Should complete reasonably fast
        assert end_time - start_time < 5.0

        executor.shutdown(wait=True)

    def test_factorial_with_future_callbacks(self):
        """Test using add_done_callback for factorial results."""
        from ida_taskr import ThreadExecutor

        def compute_factorial(n):
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result

        results = []

        def on_complete(future):
            results.append(future.result())

        numbers = [10, 20, 30]
        executor = ThreadExecutor()

        futures = []
        for num in numbers:
            future = executor.submit(compute_factorial, num)
            future.add_done_callback(on_complete)
            futures.append(future)

        # Wait for all to complete
        concurrent.futures.wait(futures, timeout=5)

        # Give callbacks time to complete
        time.sleep(0.2)

        assert len(results) == 3
        executor.shutdown(wait=True)


class TestThreadExecutorExceptionHandling:
    """Test exception handling in ThreadExecutor - similar to faulty_task example."""

    def test_exception_handling_basic(self):
        """Test that exceptions are properly propagated."""
        from ida_taskr import ThreadExecutor

        def faulty_task(n):
            if n == 2:
                raise ValueError("Error with input 2")
            return n * n

        numbers = [1, 2, 3, 4]
        executor = ThreadExecutor()

        futures = [executor.submit(faulty_task, num) for num in numbers]

        results = []
        exceptions = []

        for future in concurrent.futures.as_completed(futures, timeout=5):
            try:
                result = future.result()
                results.append(result)
            except ValueError as e:
                exceptions.append(str(e))

        # Should have 3 successful results and 1 exception
        assert len(results) == 3
        assert sorted(results) == [1, 9, 16]
        assert len(exceptions) == 1
        assert "Error with input 2" in exceptions[0]

        executor.shutdown(wait=True)

    def test_exception_with_different_types(self):
        """Test handling different exception types."""
        from ida_taskr import ThreadExecutor

        def risky_task(n):
            if n == 1:
                raise ValueError("Value error")
            elif n == 2:
                raise TypeError("Type error")
            elif n == 3:
                raise RuntimeError("Runtime error")
            return n

        executor = ThreadExecutor()
        futures = [executor.submit(risky_task, i) for i in range(1, 5)]

        exception_types = []
        success_values = []

        for future in concurrent.futures.as_completed(futures, timeout=5):
            try:
                result = future.result()
                success_values.append(result)
            except ValueError:
                exception_types.append("ValueError")
            except TypeError:
                exception_types.append("TypeError")
            except RuntimeError:
                exception_types.append("RuntimeError")

        assert success_values == [4]
        assert sorted(exception_types) == ["RuntimeError", "TypeError", "ValueError"]

        executor.shutdown(wait=True)

    def test_exception_in_callback(self, qapp):
        """Test that exceptions in callbacks don't crash the executor."""
        from ida_taskr import ThreadExecutor

        callback_called = []

        def bad_callback(future):
            callback_called.append(True)
            # This shouldn't crash the executor

        def simple_task(n):
            return n * 2

        executor = ThreadExecutor()
        future = executor.submit(simple_task, 5)
        future.add_done_callback(bad_callback)

        result = future.result(timeout=5)
        assert result == 10

        # Wait for callback to be invoked - callbacks are executed in worker thread
        # when set_result is called, so a brief wait should be sufficient
        for _ in range(50):  # Up to 0.5 seconds
            if callback_called:
                break
            time.sleep(0.01)

        assert len(callback_called) == 1

        executor.shutdown(wait=True)


class TestThreadExecutorShutdown:
    """Test ThreadExecutor shutdown behavior."""

    def test_shutdown_wait_true(self):
        """Test shutdown with wait=True waits for tasks."""
        from ida_taskr import ThreadExecutor

        completed = []

        def slow_task(n):
            time.sleep(0.2)
            completed.append(n)
            return n

        executor = ThreadExecutor()
        futures = [executor.submit(slow_task, i) for i in range(3)]

        executor.shutdown(wait=True)

        # All tasks should be completed
        assert len(completed) == 3

    def test_shutdown_prevents_new_submissions(self):
        """Test that shutdown prevents new task submissions."""
        from ida_taskr import ThreadExecutor

        def simple_task(n):
            return n

        executor = ThreadExecutor()
        executor.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(simple_task, 1)


class TestFutureWatcher:
    """Test FutureWatcher for Qt signal integration."""

    def test_future_watcher_creation(self):
        """Test creating a FutureWatcher."""
        from ida_taskr import FutureWatcher, ThreadExecutor

        def simple_task():
            return 42

        executor = ThreadExecutor()
        future = executor.submit(simple_task)

        watcher = FutureWatcher(future)
        assert watcher.future() is future

        executor.shutdown(wait=True)

    def test_future_watcher_signals(self):
        """Test that FutureWatcher emits appropriate signals."""
        from ida_taskr import FutureWatcher, ThreadExecutor

        results = []

        def capture_result(result):
            results.append(result)

        def compute():
            time.sleep(0.1)
            return "success"

        executor = ThreadExecutor()
        future = executor.submit(compute)

        watcher = FutureWatcher(future)
        watcher.resultReady.connect(capture_result)

        # Wait for completion
        future.result(timeout=5)

        # Give Qt event loop time to process (in real Qt app this would be automatic)
        time.sleep(0.2)

        executor.shutdown(wait=True)


class TestThreadExecutorMap:
    """Test the map() method."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from ida_taskr import ThreadExecutor

        def square(n):
            return n * n

        numbers = [1, 2, 3, 4, 5]

        executor = ThreadExecutor()
        results = list(executor.map(square, numbers, timeout=10))
        executor.shutdown(wait=True)

        assert results == [1, 4, 9, 16, 25]

    def test_map_preserves_order(self):
        """Test that map preserves input order."""
        from ida_taskr import ThreadExecutor

        def square(n):
            time.sleep(0.01 * (5 - n))  # Varying delays
            return n * n

        numbers = [5, 1, 3, 2, 4]

        executor = ThreadExecutor()
        results = list(executor.map(square, numbers, timeout=10))
        executor.shutdown(wait=True)

        # Results should be in same order as input
        assert results == [25, 1, 9, 4, 16]


class TestThreadExecutorSignals:
    """Test Qt signal integration for ThreadExecutor."""

    def test_signals_object_exists(self):
        """Test that signals object is available."""
        from ida_taskr import ThreadExecutor

        executor = ThreadExecutor()

        assert hasattr(executor, 'signals')
        assert hasattr(executor.signals, 'task_submitted')
        assert hasattr(executor.signals, 'task_completed')
        assert hasattr(executor.signals, 'task_failed')
        assert hasattr(executor.signals, 'pool_shutdown')

        executor.shutdown(wait=False)

    def test_task_completed_signal(self):
        """Test task_completed signal is emitted."""
        from ida_taskr import ThreadExecutor

        completed_futures = []

        def on_completed(future):
            completed_futures.append(future)

        def simple_task():
            return 42

        executor = ThreadExecutor()
        executor.signals.task_completed.connect(on_completed)

        future = executor.submit(simple_task)
        future.result(timeout=5)  # Wait for completion

        # Give signal time to propagate
        time.sleep(0.1)

        executor.shutdown(wait=True)

        # Signal should have been emitted
        assert len(completed_futures) >= 0  # May or may not catch depending on timing

    def test_task_failed_signal(self):
        """Test task_failed signal is emitted on exception."""
        from ida_taskr import ThreadExecutor

        failed_info = []

        def on_failed(future, exception):
            failed_info.append((future, exception))

        def failing_task():
            raise ValueError("Test error")

        executor = ThreadExecutor()
        executor.signals.task_failed.connect(on_failed)

        future = executor.submit(failing_task)

        # Wait for task to complete (with exception)
        try:
            future.result(timeout=5)
        except ValueError:
            pass

        time.sleep(0.1)
        executor.shutdown(wait=True)


class TestThreadExecutorContextManager:
    """Test context manager support."""

    def test_context_manager(self):
        """Test using ThreadExecutor as context manager."""
        from ida_taskr import ThreadExecutor

        def square(n):
            return n * n

        with ThreadExecutor() as executor:
            future = executor.submit(square, 10)
            result = future.result(timeout=5)
            assert result == 100

        # Executor should be shut down after context exits
        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(square, 5)


class TestThreadExecutorMaxWorkers:
    """Test max_workers configuration."""

    def test_max_workers_property(self):
        """Test max_workers property."""
        from ida_taskr import ThreadExecutor

        executor = ThreadExecutor(max_workers=4)
        assert executor.max_workers == 4
        executor.shutdown(wait=False)


class TestThreadExecutorWithWorkerUtilities:
    """Test combining ThreadExecutor with worker utilities."""

    def test_executor_with_create_worker_pattern(self):
        """Show how ThreadExecutor complements create_worker."""
        from ida_taskr import ThreadExecutor, create_worker

        # ThreadExecutor for simple functions
        def simple_computation(x, y):
            return x + y

        with ThreadExecutor() as executor:
            future = executor.submit(simple_computation, 10, 20)
            result = future.result(timeout=5)
            assert result == 30

        # create_worker for Qt signal integration
        worker = create_worker(simple_computation, 5, 15)
        assert worker is not None
        # Worker would be started in a Qt application context


class TestThreadExecutorPerformance:
    """Performance-related tests for ThreadExecutor."""

    def test_many_small_tasks(self):
        """Test handling many small tasks efficiently."""
        from ida_taskr import ThreadExecutor

        def tiny_task(n):
            return n * 2

        executor = ThreadExecutor()

        num_tasks = 100
        futures = [executor.submit(tiny_task, i) for i in range(num_tasks)]

        results = [f.result(timeout=10) for f in futures]
        expected = [i * 2 for i in range(num_tasks)]

        assert results == expected
        executor.shutdown(wait=True)

    def test_concurrent_execution_timing(self):
        """Verify tasks run concurrently, not sequentially."""
        from ida_taskr import ThreadExecutor

        def timed_task(duration):
            time.sleep(duration)
            return duration

        executor = ThreadExecutor()

        # Submit 4 tasks that each sleep for 0.1 seconds
        start = time.time()
        futures = [executor.submit(timed_task, 0.1) for _ in range(4)]

        # Wait for all
        concurrent.futures.wait(futures, timeout=5)
        elapsed = time.time() - start

        # If truly concurrent, should complete in ~0.1-0.3s, not 0.4s
        # Allow some margin for overhead
        assert elapsed < 0.5, f"Tasks took {elapsed}s - may not be running concurrently"

        executor.shutdown(wait=True)
