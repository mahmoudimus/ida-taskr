"""
SharedMemoryExecutor with Qt signals for progress tracking.

This example demonstrates how to use Qt signals to track progress
as chunks are processed.
"""

from ida_taskr import SharedMemoryExecutor
import sys


def analyze_chunk(data):
    """Pure function that analyzes a chunk of data."""
    # Simulate some analysis
    result = {
        'size': len(data),
        'high_bytes': sum(1 for b in data if b > 0x80),
        'low_bytes': sum(1 for b in data if b < 0x40),
        'checksum': sum(data) % 256
    }
    return result


def main():
    print("SharedMemoryExecutor with Qt Signals\n")

    # Create test data
    data_size = 8 * 1024 * 1024  # 8MB
    binary_data = bytearray(data_size)

    # Fill with some patterns
    for i in range(0, data_size, 100):
        binary_data[i] = 0xFF if i % 200 == 0 else 0x10

    print(f"Processing {len(binary_data)} bytes in 8 chunks...\n")

    # Create executor
    executor = SharedMemoryExecutor(max_workers=8)

    # Track progress
    chunks_completed = [0]  # Use list to allow modification in closure

    # Connect to signals
    def on_chunk_completed(chunk_id, result):
        """Called when each chunk completes."""
        chunks_completed[0] += 1
        print(f"✓ Chunk {chunk_id} completed ({chunks_completed[0]}/8)")
        print(f"  Size: {result['size']} bytes")
        print(f"  High bytes: {result['high_bytes']}")
        print(f"  Checksum: {result['checksum']}")

    def on_all_completed(combined_result):
        """Called when all chunks complete."""
        print("\n✅ All chunks completed!")
        print(f"Total results: {len(combined_result)}")

        # Calculate totals
        total_high = sum(r['high_bytes'] for r in combined_result)
        total_low = sum(r['low_bytes'] for r in combined_result)
        print(f"Total high bytes: {total_high}")
        print(f"Total low bytes: {total_low}")

    def on_task_failed(future, exception):
        """Called if processing fails."""
        print(f"❌ Task failed: {exception}")

    # Connect signals
    executor.signals.chunk_completed.connect(on_chunk_completed)
    executor.signals.all_chunks_completed.connect(on_all_completed)
    executor.signals.task_failed.connect(on_task_failed)

    try:
        # Submit chunked processing
        future = executor.submit_chunked(
            analyze_chunk,
            bytes(binary_data),
            num_chunks=8
        )

        # Wait for completion
        results = future.result()

        print("\n" + "=" * 60)
        print("Final Results")
        print("=" * 60)
        for i, result in enumerate(results):
            print(f"Chunk {i}: {result['high_bytes']} high bytes, checksum={result['checksum']}")

    finally:
        executor.shutdown(wait=True)

    print("\n✅ Processing complete with Qt signal tracking!")


def run_in_qt_app():
    """
    Example of running in a Qt application with event loop.

    In IDA or a Qt application, signals will fire through the event loop
    and can update UI elements safely.
    """
    from ida_taskr.qt_compat import QtCore, QT_AVAILABLE

    if not QT_AVAILABLE:
        print("Qt not available - skipping Qt app example")
        return

    QCoreApplication = QtCore.QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    # Create test data
    data_size = 4 * 1024 * 1024  # 4MB
    binary_data = bytes(data_size)

    # Create executor
    executor = SharedMemoryExecutor(max_workers=4)

    # Track completion
    completed = [False]

    def on_completed(result):
        print(f"✅ Completed! Processed {len(result)} chunks")
        completed[0] = True
        app.quit()

    executor.signals.all_chunks_completed.connect(on_completed)

    # Submit work
    print("Submitting work in Qt event loop...")
    future = executor.submit_chunked(
        analyze_chunk,
        binary_data,
        num_chunks=4
    )

    # Run event loop (in IDA, this isn't needed as event loop already runs)
    # app.exec_()

    # For this example, just wait synchronously
    future.result()
    on_completed([])

    executor.shutdown(wait=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Simple signal tracking")
    print("=" * 60)
    main()

    print("\n\n" + "=" * 60)
    print("Example 2: Qt event loop integration")
    print("=" * 60)
    run_in_qt_app()
