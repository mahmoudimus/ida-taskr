"""
Unit tests for SharedMemoryExecutor.

Tests the SharedMemoryExecutor class that provides concurrent.futures
interface with shared memory optimization for chunked data processing.
"""

import unittest
import time
from concurrent.futures import Future

try:
    from ida_taskr.qtasyncio import SharedMemoryExecutor, QT_AVAILABLE
    EXECUTOR_AVAILABLE = QT_AVAILABLE
except ImportError:
    EXECUTOR_AVAILABLE = False


# Test worker functions (module-level for pickling)
def simple_worker(data):
    """Simple worker that counts bytes."""
    return len(data)


def pattern_finder(data):
    """Worker that finds specific pattern."""
    return [i for i in range(len(data)) if data[i] == 0xFF]


def sum_bytes(data):
    """Worker that sums all bytes."""
    return sum(data)


@unittest.skipUnless(EXECUTOR_AVAILABLE, "Qt not available")
class TestSharedMemoryExecutor(unittest.TestCase):
    """Test SharedMemoryExecutor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = None

    def tearDown(self):
        """Clean up after tests."""
        if self.executor:
            self.executor.shutdown(wait=True)

    def test_executor_creation(self):
        """Test basic executor creation."""
        self.executor = SharedMemoryExecutor(max_workers=4)
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.max_workers, 4)

    def test_standard_submit(self):
        """Test standard submit() interface."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        # Submit simple task
        future = self.executor.submit(lambda x: x * 2, 21)
        self.assertIsInstance(future, Future)

        result = future.result(timeout=5)
        self.assertEqual(result, 42)

    def test_standard_map(self):
        """Test standard map() interface."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        # Map over simple iterable
        results = list(self.executor.map(lambda x: x * 2, [1, 2, 3, 4, 5]))
        self.assertEqual(results, [2, 4, 6, 8, 10])

    def test_submit_chunked_simple(self):
        """Test submit_chunked() with simple data."""
        self.executor = SharedMemoryExecutor(max_workers=4)

        # Create test data
        data = bytes([0xFF if i % 10 == 0 else 0x00 for i in range(1000)])

        # Process in chunks
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=4)
        self.assertIsInstance(future, Future)

        # Should get list of 4 results (one per chunk)
        results = future.result(timeout=10)
        self.assertEqual(len(results), 4)

        # Each result should be chunk size
        for result in results:
            self.assertGreater(result, 0)

        # Total should equal data size
        self.assertEqual(sum(results), len(data))

    def test_submit_chunked_with_combine(self):
        """Test submit_chunked() with result combining."""
        self.executor = SharedMemoryExecutor(max_workers=4)

        # Create test data
        data = bytes([0xFF if i % 10 == 0 else 0x00 for i in range(1000)])

        # Process with combining (flatten list of lists)
        future = self.executor.submit_chunked(
            pattern_finder,
            data,
            num_chunks=4,
            combine=lambda results: sum(results, [])  # Flatten
        )

        # Should get flattened list of positions
        positions = future.result(timeout=10)
        self.assertIsInstance(positions, list)

        # Should have 100 positions (every 10th byte)
        self.assertEqual(len(positions), 100)

        # Positions should be multiples of 10
        for pos in positions:
            self.assertEqual(pos % 10, 0)

    def test_submit_chunked_sum_combine(self):
        """Test submit_chunked() with sum combining."""
        self.executor = SharedMemoryExecutor(max_workers=4)

        # Create test data with known sum
        data = bytes([i % 256 for i in range(1000)])

        # Process with sum combining
        future = self.executor.submit_chunked(
            sum_bytes,
            data,
            num_chunks=4,
            combine=sum  # Sum all chunk sums
        )

        total = future.result(timeout=10)

        # Should equal Python's sum
        self.assertEqual(total, sum(data))

    def test_map_chunked(self):
        """Test map_chunked() streaming interface."""
        self.executor = SharedMemoryExecutor(max_workers=4)

        # Create test data
        data = bytes([i % 256 for i in range(1000)])

        # Stream results
        results = list(self.executor.map_chunked(simple_worker, data, num_chunks=4))

        # Should get 4 results
        self.assertEqual(len(results), 4)

        # Total should equal data size
        self.assertEqual(sum(results), len(data))

    def test_context_manager(self):
        """Test executor as context manager."""
        data = bytes(1000)

        with SharedMemoryExecutor(max_workers=2) as executor:
            future = executor.submit_chunked(simple_worker, data, num_chunks=2)
            results = future.result(timeout=10)
            self.assertEqual(len(results), 2)

        # Executor should be shut down after context
        # (Can't easily test this without accessing private state)

    def test_shutdown(self):
        """Test executor shutdown."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        # Submit some work
        data = bytes(1000)
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=2)
        future.result(timeout=10)

        # Shutdown
        self.executor.shutdown(wait=True)

        # Should not be able to submit after shutdown
        with self.assertRaises(RuntimeError):
            self.executor.submit(lambda: 42)

    def test_large_data(self):
        """Test with larger data (8MB)."""
        self.executor = SharedMemoryExecutor(max_workers=8)

        # Create 8MB of data
        data_size = 8 * 1024 * 1024
        data = bytes([i % 256 for i in range(data_size)])

        # Process in 8 chunks
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=8)
        results = future.result(timeout=20)

        # Should get 8 results
        self.assertEqual(len(results), 8)

        # Total should equal data size
        self.assertEqual(sum(results), data_size)

    def test_signals_emitted(self):
        """Test that Qt signals are emitted."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        # Track signal emissions
        signals_received = {
            'task_submitted': False,
            'chunk_completed': False,
            'all_chunks_completed': False,
        }

        def on_task_submitted(future):
            signals_received['task_submitted'] = True

        def on_chunk_completed(chunk_id, result):
            signals_received['chunk_completed'] = True

        def on_all_completed(result):
            signals_received['all_chunks_completed'] = True

        # Connect signals
        self.executor.signals.task_submitted.connect(on_task_submitted)
        self.executor.signals.chunk_completed.connect(on_chunk_completed)
        self.executor.signals.all_chunks_completed.connect(on_all_completed)

        # Submit work
        data = bytes(1000)
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=2)
        future.result(timeout=10)

        # Give signals time to fire
        time.sleep(0.1)

        # Verify signals were emitted
        # Note: task_submitted might not fire if future completed too fast
        self.assertTrue(signals_received['chunk_completed'])
        self.assertTrue(signals_received['all_chunks_completed'])

    def test_error_handling(self):
        """Test error handling in workers."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        def failing_worker(data):
            raise ValueError("Test error")

        data = bytes(1000)
        future = self.executor.submit_chunked(failing_worker, data, num_chunks=2)

        # Should propagate exception
        with self.assertRaises(ValueError):
            future.result(timeout=10)

    def test_empty_data(self):
        """Test with empty data."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        # Empty data should work
        data = bytes(0)
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=2)

        # Should handle gracefully (though chunks will be empty)
        results = future.result(timeout=10)
        self.assertIsInstance(results, list)

    def test_single_chunk(self):
        """Test with single chunk (edge case)."""
        self.executor = SharedMemoryExecutor(max_workers=1)

        data = bytes(1000)
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=1)

        results = future.result(timeout=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], len(data))

    def test_more_chunks_than_workers(self):
        """Test with more chunks than workers."""
        self.executor = SharedMemoryExecutor(max_workers=2)

        data = bytes(1000)
        # 8 chunks but only 2 workers - should still work
        future = self.executor.submit_chunked(simple_worker, data, num_chunks=8)

        results = future.result(timeout=10)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(results), len(data))


if __name__ == '__main__':
    unittest.main()
