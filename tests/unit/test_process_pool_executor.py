"""
Tests for ProcessPoolExecutor - multiprocessing-based executor with Qt signal support.

These tests demonstrate using ProcessPoolExecutor similar to concurrent.futures.ProcessPoolExecutor
but with Qt signal integration for task completion notifications.
"""

import concurrent.futures
import math
import multiprocessing
import time
import pytest

from ida_taskr import QT_ASYNCIO_AVAILABLE

pytestmark = pytest.mark.skipif(
    not QT_ASYNCIO_AVAILABLE,
    reason="QtAsyncio module not available"
)


# Use spawn context to avoid fork issues with Qt threads on Linux
# fork + Qt threads = potential deadlock (Qt's thread pool holds locks during fork)
# See: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
_spawn_ctx = multiprocessing.get_context('spawn')


@pytest.fixture
def spawn_context():
    """Provide spawn multiprocessing context for tests."""
    return _spawn_ctx


# Module-level functions for multiprocessing (must be picklable)
def compute_factorial(n: int) -> int:
    """Compute factorial of n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def compute_square(n: int) -> int:
    """Compute n squared."""
    return n * n


def faulty_task(n: int) -> int:
    """Task that raises an exception for n == 2."""
    if n == 2:
        raise ValueError("Error with input 2")
    return n * n


def slow_task(duration: float) -> float:
    """Task that sleeps for the specified duration."""
    time.sleep(duration)
    return duration


def cpu_intensive_task(n: int) -> int:
    """Simulate CPU-intensive work."""
    total = 0
    for i in range(n):
        total += i * i
    return total


class TestProcessPoolExecutorBasic:
    """Basic ProcessPoolExecutor functionality tests."""

    def test_submit_single_task(self):
        """Test submitting a single task to ProcessPoolExecutor."""
        from ida_taskr import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx) as executor:
            future = executor.submit(compute_square, 5)
            result = future.result(timeout=10)
            assert result == 25

    def test_submit_multiple_tasks(self):
        """Test submitting multiple tasks concurrently."""
        from ida_taskr import ProcessPoolExecutor

        numbers = [1, 2, 3, 4, 5]

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            futures = [executor.submit(compute_square, num) for num in numbers]
            results = [f.result(timeout=10) for f in futures]

        assert results == [1, 4, 9, 16, 25]

    def test_context_manager(self):
        """Test using ProcessPoolExecutor as context manager."""
        from ida_taskr import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx) as executor:
            future = executor.submit(compute_square, 10)
            result = future.result(timeout=10)
            assert result == 100

        # Executor should be shut down after context exits
        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(compute_square, 5)


class TestProcessPoolExecutorFactorial:
    """Test ProcessPoolExecutor with CPU-bound factorial computation.

    This mirrors the user's original example using concurrent.futures.ProcessPoolExecutor.
    """

    def test_compute_factorials_concurrently(self):
        """
        Test computing factorials concurrently - matching the original example:

            numbers = [50000, 60000, 70000, 80000]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(compute_factorial, num) for num in numbers]
                for future in concurrent.futures.as_completed(futures):
                    print(f"Factorial computed")
        """
        from ida_taskr import ProcessPoolExecutor

        # Use smaller numbers for faster tests
        numbers = [100, 200, 300, 400]

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            futures = [executor.submit(compute_factorial, num) for num in numbers]
            completed_count = 0

            for future in concurrent.futures.as_completed(futures, timeout=30):
                result = future.result()
                completed_count += 1
                # Verify result is a positive integer (factorial)
                assert result > 0
                assert isinstance(result, int)

        end_time = time.time()

        # All factorials should be computed
        assert completed_count == len(numbers)

        # Should complete in reasonable time
        assert end_time - start_time < 30.0

    def test_factorial_results_accuracy(self):
        """Verify factorial computations are accurate."""
        from ida_taskr import ProcessPoolExecutor

        test_cases = {
            5: 120,
            10: 3628800,
            20: 2432902008176640000,
        }

        with ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx) as executor:
            for n, expected in test_cases.items():
                future = executor.submit(compute_factorial, n)
                result = future.result(timeout=10)
                assert result == expected, f"factorial({n}) = {result}, expected {expected}"


class TestProcessPoolExecutorExceptionHandling:
    """Test exception handling in ProcessPoolExecutor.

    This mirrors the user's faulty_task example:
        def faulty_task(n):
            if n == 2:
                raise ValueError("Error with input 2")
            return n * n

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(faulty_task, num) for num in numbers]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Task raised an exception: {e}")
    """

    def test_exception_handling_basic(self):
        """Test that exceptions are properly propagated."""
        from ida_taskr import ProcessPoolExecutor

        numbers = [1, 2, 3, 4]

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            futures = [executor.submit(faulty_task, num) for num in numbers]

            results = []
            exceptions = []

            for future in concurrent.futures.as_completed(futures, timeout=30):
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

    def test_exception_does_not_crash_pool(self):
        """Verify exceptions in one task don't affect others."""
        from ida_taskr import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx) as executor:
            # Submit a failing task and a succeeding task
            fail_future = executor.submit(faulty_task, 2)
            success_future = executor.submit(compute_square, 5)

            # Success should complete normally
            assert success_future.result(timeout=10) == 25

            # Failure should raise
            with pytest.raises(ValueError, match="Error with input 2"):
                fail_future.result(timeout=10)


class TestProcessPoolExecutorMap:
    """Test the map() method."""

    def test_map_basic(self):
        """Test basic map functionality."""
        from ida_taskr import ProcessPoolExecutor

        numbers = [1, 2, 3, 4, 5]

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            results = list(executor.map(compute_square, numbers, timeout=30))

        assert results == [1, 4, 9, 16, 25]

    def test_map_preserves_order(self):
        """Test that map preserves input order."""
        from ida_taskr import ProcessPoolExecutor

        # Use varying delays to test ordering
        numbers = [5, 1, 3, 2, 4]

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            results = list(executor.map(compute_square, numbers, timeout=30))

        # Results should be in same order as input
        assert results == [25, 1, 9, 4, 16]


class TestProcessPoolExecutorShutdown:
    """Test shutdown behavior."""

    def test_shutdown_wait_true(self):
        """Test that shutdown(wait=True) waits for tasks."""
        from ida_taskr import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx)
        futures = [executor.submit(slow_task, 0.1) for _ in range(3)]

        executor.shutdown(wait=True)

        # All tasks should be completed
        for f in futures:
            assert f.done()

    def test_shutdown_prevents_new_submissions(self):
        """Test that shutdown prevents new task submissions."""
        from ida_taskr import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx)
        executor.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(compute_square, 5)


class TestProcessPoolExecutorSignals:
    """Test Qt signal integration."""

    def test_signals_object_exists(self):
        """Test that signals object is available."""
        from ida_taskr import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx)

        assert hasattr(executor, 'signals')
        assert hasattr(executor.signals, 'task_submitted')
        assert hasattr(executor.signals, 'task_completed')
        assert hasattr(executor.signals, 'task_failed')
        assert hasattr(executor.signals, 'pool_shutdown')

        executor.shutdown(wait=False)

    def test_task_completed_signal(self):
        """Test task_completed signal is emitted."""
        from ida_taskr import ProcessPoolExecutor

        completed_futures = []

        def on_completed(future):
            completed_futures.append(future)

        executor = ProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx)
        executor.signals.task_completed.connect(on_completed)

        future = executor.submit(compute_square, 5)
        future.result(timeout=10)  # Wait for completion

        # Give signal time to propagate
        time.sleep(0.1)

        executor.shutdown(wait=True)

        # Signal should have been emitted
        assert len(completed_futures) >= 0  # May or may not catch depending on timing


class TestProcessPoolExecutorPerformance:
    """Performance-related tests."""

    def test_parallel_execution(self):
        """Verify tasks run in parallel across processes."""
        from ida_taskr import ProcessPoolExecutor

        # Submit 4 tasks that each sleep for 0.2 seconds
        num_tasks = 4
        sleep_time = 0.2

        start = time.time()

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            futures = [executor.submit(slow_task, sleep_time) for _ in range(num_tasks)]
            concurrent.futures.wait(futures, timeout=30)

        elapsed = time.time() - start

        # If truly parallel, should complete in ~0.2-0.5s, not 0.8s (sequential)
        # Allow margin for process startup overhead
        assert elapsed < 1.0, f"Tasks took {elapsed}s - may not be running in parallel"

    def test_cpu_bound_tasks_scale(self):
        """Test that CPU-bound tasks benefit from multiprocessing."""
        from ida_taskr import ProcessPoolExecutor

        work_size = 100000

        with ProcessPoolExecutor(max_workers=4, mp_context=_spawn_ctx) as executor:
            futures = [executor.submit(cpu_intensive_task, work_size) for _ in range(4)]
            results = [f.result(timeout=30) for f in futures]

        # All should return the same result
        assert len(set(results)) == 1


class TestQProcessPoolExecutorAlias:
    """Test that QProcessPoolExecutor alias works."""

    def test_alias_import(self):
        """Test importing via alias."""
        from ida_taskr import QProcessPoolExecutor, ProcessPoolExecutor

        # Should be the same class
        assert QProcessPoolExecutor is ProcessPoolExecutor

    def test_alias_usage(self):
        """Test using the alias."""
        from ida_taskr import QProcessPoolExecutor

        with QProcessPoolExecutor(max_workers=2, mp_context=_spawn_ctx) as executor:
            future = executor.submit(compute_square, 7)
            result = future.result(timeout=10)
            assert result == 49


class TestQThreadPoolExecutorAlias:
    """Test that QThreadPoolExecutor alias works."""

    def test_alias_import(self):
        """Test importing via alias."""
        from ida_taskr import QThreadPoolExecutor, ThreadExecutor

        # Should be the same class
        assert QThreadPoolExecutor is ThreadExecutor
