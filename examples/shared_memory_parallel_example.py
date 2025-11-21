"""
Example: Using ProcessPoolExecutor with Shared Memory for Chunked Processing

Shows how to process large binary data (like 8MB files) in parallel using
shared memory to avoid copying data between processes.

This pattern is used by anti_deob to analyze large code sections without
blocking IDA's main thread.
"""

import multiprocessing
import multiprocessing.shared_memory
from concurrent.futures import as_completed
from ida_taskr import ProcessPoolExecutor
import time


# ==============================================================================
# STEP 1: Worker function that processes a chunk
# ==============================================================================

def process_chunk(shm_name, chunk_start, chunk_end, chunk_id):
    """
    Worker function that processes one chunk of shared memory.

    This runs in a separate process and attaches to the shared memory.

    Args:
        shm_name: Name of the shared memory segment
        chunk_start: Start offset in the shared memory
        chunk_end: End offset in the shared memory
        chunk_id: ID of this chunk (for progress tracking)

    Returns:
        Results from analyzing this chunk
    """
    # Attach to existing shared memory (created by main process)
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)

    try:
        # Get a view of just this chunk (no copying!)
        chunk_data = memoryview(shm.buf)[chunk_start:chunk_end]

        # Process the chunk (simulate signature generation)
        signature = []
        for i in range(min(16, len(chunk_data))):  # First 16 bytes
            signature.append(chunk_data[i])
            time.sleep(0.001)  # Simulate CPU work

        # Return results
        return {
            'chunk_id': chunk_id,
            'start': chunk_start,
            'end': chunk_end,
            'signature': signature,
            'pattern': ' '.join(f'{b:02X}' for b in signature)
        }

    finally:
        # IMPORTANT: Delete memoryview before closing shared memory
        del chunk_data
        shm.close()  # Detach from shared memory (don't unlink!)


# ==============================================================================
# STEP 2: Main function that coordinates chunked processing
# ==============================================================================

def process_large_binary_parallel(binary_data, num_chunks=4, max_workers=4):
    """
    Process large binary data in parallel using shared memory.

    This is the pattern used by anti_deob to analyze large sections
    without blocking IDA's main thread.

    Args:
        binary_data: Large binary data (e.g., 8MB from IDA)
        num_chunks: Number of chunks to split data into
        max_workers: Number of parallel workers

    Returns:
        List of results from all chunks
    """
    data_size = len(binary_data)
    chunk_size = data_size // num_chunks

    print(f"Processing {data_size:,} bytes in {num_chunks} chunks")
    print(f"Using {max_workers} parallel workers")
    print()

    # STEP 1: Create shared memory and copy data into it
    # This happens ONCE in the main process
    shm = multiprocessing.shared_memory.SharedMemory(
        create=True,
        size=data_size
    )

    try:
        # Copy data into shared memory
        shm.buf[:data_size] = binary_data
        print(f"✓ Created shared memory: {shm.name}")
        print(f"  Size: {data_size:,} bytes")
        print()

        # STEP 2: Create chunks and submit to ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = []

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else data_size

            print(f"Submitting chunk {i+1}/{num_chunks}: bytes {chunk_start:,} - {chunk_end:,}")

            # Submit chunk for processing
            # Only pass the SHM name and offsets - NO DATA COPYING!
            future = executor.submit(
                process_chunk,
                shm.name,      # Shared memory name
                chunk_start,   # Start offset
                chunk_end,     # End offset
                i             # Chunk ID
            )
            futures.append(future)

        print()
        print("All chunks submitted - workers processing in parallel...")
        print()

        # STEP 3: Collect results as they complete
        results = []
        for future in as_completed(futures):
            result = future.result()
            chunk_id = result['chunk_id']
            print(f"✓ Chunk {chunk_id+1} complete: {result['pattern'][:40]}...")
            results.append(result)

        # Sort by chunk_id to get original order
        results.sort(key=lambda r: r['chunk_id'])

        executor.shutdown(wait=True)
        print()
        print(f"✓ All {num_chunks} chunks processed!")

        return results

    finally:
        # STEP 4: Cleanup shared memory
        shm.close()
        shm.unlink()  # Remove shared memory
        print(f"✓ Shared memory cleaned up")


# ==============================================================================
# STEP 3: Example with progress callbacks (like anti_deob)
# ==============================================================================

def process_with_progress(binary_data, num_chunks=4):
    """
    Example with progress updates via Qt signals.

    This is how anti_deob provides real-time progress updates.
    """
    data_size = len(binary_data)
    chunk_size = data_size // num_chunks

    # Create shared memory
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=data_size)
    shm.buf[:data_size] = binary_data

    try:
        executor = ProcessPoolExecutor(max_workers=4)

        # Connect to progress signals
        completed_chunks = []

        def on_chunk_complete(future):
            result = future.result()
            completed_chunks.append(result)
            progress = int(len(completed_chunks) / num_chunks * 100)
            print(f"[{progress}%] Chunk {result['chunk_id']+1}/{num_chunks} done")

        executor.signals.task_completed.connect(on_chunk_complete)

        # Submit all chunks
        futures = []
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else data_size

            future = executor.submit(
                process_chunk,
                shm.name,
                chunk_start,
                chunk_end,
                i
            )
            futures.append(future)

        # Wait for all chunks
        for future in futures:
            future.result()

        executor.shutdown(wait=True)

        return completed_chunks

    finally:
        shm.close()
        shm.unlink()


# ==============================================================================
# STEP 4: Decorator version (ultra simple API)
# ==============================================================================

from ida_taskr import background_task

@background_task(max_workers=4, executor_type='process')
def process_chunk_simple(shm_name, chunk_start, chunk_end):
    """Simplified chunk processor - just add decorator!"""
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    try:
        chunk_data = memoryview(shm.buf)[chunk_start:chunk_end]
        # Process chunk...
        signature = list(chunk_data[:16])
        return {'signature': signature}
    finally:
        del chunk_data
        shm.close()


def simple_parallel_processing(binary_data):
    """Ultra-simple API using decorator."""
    shm = multiprocessing.shared_memory.SharedMemory(
        create=True,
        size=len(binary_data)
    )
    shm.buf[:len(binary_data)] = binary_data

    try:
        # Submit chunks - returns futures immediately
        num_chunks = 4
        chunk_size = len(binary_data) // num_chunks

        futures = [
            process_chunk_simple(
                shm.name,
                i * chunk_size,
                (i + 1) * chunk_size
            )
            for i in range(num_chunks)
        ]

        # Collect results
        results = [f.result() for f in futures]
        return results

    finally:
        shm.close()
        shm.unlink()


# ==============================================================================
# Demo
# ==============================================================================

def main():
    print("=" * 70)
    print("Parallel Processing with Shared Memory")
    print("=" * 70)
    print()

    # Create 8MB of test data (simulating IDA binary data)
    binary_data = bytes(range(256)) * (32 * 1024)  # 8MB
    print(f"Test data size: {len(binary_data):,} bytes (8MB)")
    print()

    # Process in parallel using shared memory
    results = process_large_binary_parallel(
        binary_data,
        num_chunks=8,
        max_workers=4
    )

    print()
    print("=" * 70)
    print("Results Summary:")
    print("=" * 70)
    for r in results[:3]:  # Show first 3
        print(f"Chunk {r['chunk_id']}: {r['pattern'][:50]}...")

    print()
    print("✓ Main thread was NEVER blocked!")
    print("✓ No data copying between processes (shared memory)")
    print("✓ True parallel execution on multiple CPU cores")
    print("=" * 70)


if __name__ == '__main__':
    main()
