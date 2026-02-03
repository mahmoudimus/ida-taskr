"""
Tests for InterpreterPoolExecutor - process-based executor with Qt signal support.

These tests demonstrate using InterpreterPoolExecutor similar to
concurrent.futures patterns but with Qt signal integration.

This implementation uses ProcessPoolExecutor as the backend, providing
true parallelism compatible with embedded Python contexts (like IDA Pro).
"""

import concurrent.futures
import sys
import time
import pytest

from ida_taskr import QT_ASYNCIO_AVAILABLE, INTERPRETER_POOL_AVAILABLE

# Check if using PyQt5 (not PySide6)
try:
    import PyQt5
    _USING_PYQT5 = True
except ImportError:
    _USING_PYQT5 = False

# Skip on PyQt5 + Python 3.12+ due to multiprocessing/spawn compatibility issues
# that cause hangs in CI. PySide6 works fine with all Python versions.
_PYQT5_PY312_ISSUE = _USING_PYQT5 and sys.version_info >= (3, 12)

# Skip all tests if QtAsyncio is not available
pytestmark = [
    pytest.mark.skipif(
        not QT_ASYNCIO_AVAILABLE,
        reason="QtAsyncio module not available"
    ),
    pytest.mark.skipif(
        _PYQT5_PY312_ISSUE,
        reason="PyQt5 + Python 3.12+ has multiprocessing spawn issues in CI"
    ),
]


# Module-level functions for interpreter sharing (must be shareable)
def compute_sum_of_squares(num: int) -> int:
    """Compute sum of squares from 0 to num."""
    return sum(i * i for i in range(num + 1))


def compute_factorial(n: int) -> int:
    """Compute factorial of n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def faulty_task(n: int) -> int:
    """Task that raises an exception for n == 2."""
    if n == 2:
        raise ValueError("Error with input 2")
    return n * n


def slow_task(duration: float) -> float:
    """Task that sleeps for the specified duration."""
    time.sleep(duration)
    return duration


class TestInterpreterPoolAvailability:
    """Test availability detection."""

    def test_availability_flag_exists(self):
        """Test that INTERPRETER_POOL_AVAILABLE flag is exported."""
        from ida_taskr import INTERPRETER_POOL_AVAILABLE

        # Should be a boolean
        assert isinstance(INTERPRETER_POOL_AVAILABLE, bool)

        # Should always be True (uses ProcessPoolExecutor as backend)
        assert INTERPRETER_POOL_AVAILABLE is True

    def test_signals_class_always_available(self):
        """Test that the signals class is always importable."""
        from ida_taskr.qtasyncio import InterpreterPoolExecutorSignals

        # Should always be available
        assert InterpreterPoolExecutorSignals is not None

    def test_executor_class_always_available(self):
        """Test that the executor class is always importable."""
        from ida_taskr import InterpreterPoolExecutor

        # Should always be available (uses ProcessPoolExecutor backend)
        assert InterpreterPoolExecutor is not None


class TestInterpreterPoolExecutorBasic:
    """Basic InterpreterPoolExecutor functionality tests."""

    def test_submit_single_task(self):
        """Test submitting a single task to InterpreterPoolExecutor."""
        from ida_taskr import InterpreterPoolExecutor

        with InterpreterPoolExecutor(max_workers=2) as executor:
            future = executor.submit(compute_sum_of_squares, 100)
            result = future.result(timeout=10)
            expected = sum(i * i for i in range(101))
            assert result == expected

    def test_submit_multiple_tasks(self):
        """Test submitting multiple tasks concurrently."""
        from ida_taskr import InterpreterPoolExecutor

        numbers = [100, 200, 300, 400]

        with InterpreterPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_sum_of_squares, num) for num in numbers]
            results = [f.result(timeout=10) for f in futures]

        # Verify all results
        for num, result in zip(numbers, results):
            expected = sum(i * i for i in range(num + 1))
            assert result == expected

    def test_context_manager(self):
        """Test using InterpreterPoolExecutor as context manager."""
        from ida_taskr import InterpreterPoolExecutor

        with InterpreterPoolExecutor(max_workers=2) as executor:
            future = executor.submit(compute_sum_of_squares, 50)
            result = future.result(timeout=10)
            assert result > 0

        # Executor should be shut down after context exits
        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(compute_sum_of_squares, 10)


class TestInterpreterPoolExecutorMap:
    """Test the map() method - matching the user's example."""

    def test_map_sum_of_squares(self):
        """
        Test map with sum of squares - matching the user's example:

            def sums(num: int) -> int:
                return sum(i * i for i in range(num + 1))

            with InterpreterPoolExecutor() as executor:
                print(list(executor.map(sums, [100_000] * 4)))
        """
        from ida_taskr import InterpreterPoolExecutor

        # Use smaller numbers for faster tests
        numbers = [1000] * 4

        with InterpreterPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(compute_sum_of_squares, numbers, timeout=30))

        # All should return the same value
        assert len(results) == 4
        expected = sum(i * i for i in range(1001))
        assert all(r == expected for r in results)

    def test_map_preserves_order(self):
        """Test that map preserves input order."""
        from ida_taskr import InterpreterPoolExecutor

        numbers = [100, 200, 300, 400, 500]

        with InterpreterPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(compute_sum_of_squares, numbers, timeout=30))

        # Results should be in same order as input
        for num, result in zip(numbers, results):
            expected = sum(i * i for i in range(num + 1))
            assert result == expected


class TestInterpreterPoolExecutorExceptionHandling:
    """Test exception handling in InterpreterPoolExecutor."""

    def test_exception_handling_basic(self):
        """Test that exceptions are properly propagated."""
        from ida_taskr import InterpreterPoolExecutor

        numbers = [1, 2, 3, 4]

        with InterpreterPoolExecutor(max_workers=4) as executor:
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
        from ida_taskr import InterpreterPoolExecutor

        with InterpreterPoolExecutor(max_workers=2) as executor:
            # Submit a failing task and a succeeding task
            fail_future = executor.submit(faulty_task, 2)
            success_future = executor.submit(compute_sum_of_squares, 10)

            # Success should complete normally
            expected = sum(i * i for i in range(11))
            assert success_future.result(timeout=10) == expected

            # Failure should raise
            with pytest.raises(ValueError, match="Error with input 2"):
                fail_future.result(timeout=10)


class TestInterpreterPoolExecutorShutdown:
    """Test shutdown behavior."""

    def test_shutdown_wait_true(self):
        """Test that shutdown(wait=True) waits for tasks."""
        from ida_taskr import InterpreterPoolExecutor

        executor = InterpreterPoolExecutor(max_workers=2)
        futures = [executor.submit(slow_task, 0.1) for _ in range(3)]

        executor.shutdown(wait=True)

        # All tasks should be completed
        for f in futures:
            assert f.done()

    def test_shutdown_prevents_new_submissions(self):
        """Test that shutdown prevents new task submissions."""
        from ida_taskr import InterpreterPoolExecutor

        executor = InterpreterPoolExecutor(max_workers=2)
        executor.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="Cannot schedule new futures"):
            executor.submit(compute_sum_of_squares, 10)


class TestInterpreterPoolExecutorSignals:
    """Test Qt signal integration (ProcessPoolExecutor backend)."""

    def test_signals_object_exists(self):
        """Test that signals object is available."""
        from ida_taskr import InterpreterPoolExecutor

        executor = InterpreterPoolExecutor(max_workers=2)

        assert hasattr(executor, 'signals')
        assert hasattr(executor.signals, 'task_submitted')
        assert hasattr(executor.signals, 'task_completed')
        assert hasattr(executor.signals, 'task_failed')
        assert hasattr(executor.signals, 'pool_shutdown')

        executor.shutdown(wait=False)

    def test_max_workers_property(self):
        """Test max_workers property."""
        from ida_taskr import InterpreterPoolExecutor

        executor = InterpreterPoolExecutor(max_workers=4)
        assert executor.max_workers == 4
        executor.shutdown(wait=False)


class TestInterpreterPoolExecutorPerformance:
    """Performance-related tests (ProcessPoolExecutor backend)."""

    def test_parallel_execution(self):
        """Verify tasks run in parallel across processes."""
        from ida_taskr import InterpreterPoolExecutor

        # Submit 4 tasks that each sleep for 0.2 seconds
        num_tasks = 4
        sleep_time = 0.2

        start = time.time()

        with InterpreterPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(slow_task, sleep_time) for _ in range(num_tasks)]
            concurrent.futures.wait(futures, timeout=30)

        elapsed = time.time() - start

        # If truly parallel, should complete in ~0.2-0.5s, not 0.8s (sequential)
        # Allow margin for interpreter startup overhead
        assert elapsed < 1.0, f"Tasks took {elapsed}s - may not be running in parallel"


class TestQInterpreterPoolExecutorAlias:
    """Test that QInterpreterPoolExecutor alias works."""

    def test_alias_import(self):
        """Test importing via alias."""
        from ida_taskr import QInterpreterPoolExecutor, InterpreterPoolExecutor

        # Should be the same class
        assert QInterpreterPoolExecutor is InterpreterPoolExecutor

    def test_alias_usage(self):
        """Test using the alias."""
        from ida_taskr import QInterpreterPoolExecutor

        with QInterpreterPoolExecutor(max_workers=2) as executor:
            future = executor.submit(compute_sum_of_squares, 100)
            result = future.result(timeout=10)
            expected = sum(i * i for i in range(101))
            assert result == expected
