"""
Minimal Shared Memory API

Shows the smallest amount of code users need to write for shared memory tasks.

This example demonstrates the actual working @shared_memory_task decorator.
"""

import time
import sys
import os

# Add src to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ==============================================================================
# BEFORE: Manual shared memory setup (COMPLEX)
# ==============================================================================

def analyze_manual(binary_data):
    """
    Manual shared memory - lots of boilerplate!

    User has to handle:
    - Create SharedMemory
    - Copy data into it
    - Calculate chunk boundaries
    - Create worker that attaches to shared memory
    - Submit chunks with offsets
    - Cleanup memoryview
    - Close/unlink shared memory

    ~40+ lines of code!
    """
    import multiprocessing
    from multiprocessing import shared_memory
    from ida_taskr import ProcessPoolExecutor

    # Step 1: Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=len(binary_data))
    shm.buf[:] = binary_data

    # Step 2: Calculate chunks
    num_chunks = 8
    chunk_size = len(binary_data) // num_chunks

    # Step 3: Define worker that attaches to shared memory
    def worker(shm_name, start, end, chunk_id):
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            # Attach to shared memory
            chunk_data = memoryview(shm.buf)[start:end]

            # Process chunk
            result = []
            for i in range(min(16, len(chunk_data))):
                result.append(chunk_data[i])

            return {'chunk_id': chunk_id, 'signature': result}
        finally:
            del chunk_data  # CRITICAL!
            shm.close()

    # Step 4: Submit chunks
    executor = ProcessPoolExecutor(max_workers=8)
    futures = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(binary_data)
        future = executor.submit(worker, shm.name, start, end, i)
        futures.append(future)

    # Step 5: Collect results
    results = [f.result() for f in futures]

    # Step 6: Cleanup
    executor.shutdown()
    shm.close()
    shm.unlink()

    return results


# ==============================================================================
# AFTER: With decorator (MINIMAL)
# ==============================================================================

from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze_minimal(chunk_data, chunk_id, total_chunks):
    """
    Minimal shared memory API!

    User just writes the chunk processing logic.
    ida-taskr handles:
    - Creating/managing shared memory
    - Chunking the data
    - Worker attachment
    - Cleanup

    Just ~5 lines of actual logic!
    """
    # Process this chunk
    result = []
    for i in range(min(16, len(chunk_data))):
        result.append(chunk_data[i])

    return {'chunk_id': chunk_id, 'signature': result}


# Usage:
# results = analyze_minimal(binary_data)  # That's it!
# ida-taskr handles all the shared memory complexity


# ==============================================================================
# Comparison
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SHARED MEMORY: Minimal API")
    print("=" * 70)
    print()

    print("WITHOUT decorator (manual):")
    print("-" * 70)
    print("""
def analyze(binary_data):
    import multiprocessing.shared_memory

    # 1. Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=len(binary_data))
    shm.buf[:] = binary_data

    # 2. Calculate chunks
    num_chunks = 8
    chunk_size = len(binary_data) // num_chunks

    # 3. Define worker with shared memory attachment
    def worker(shm_name, start, end, chunk_id):
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            chunk_data = memoryview(shm.buf)[start:end]
            # ... process chunk ...
            return result
        finally:
            del chunk_data
            shm.close()

    # 4. Submit chunks with offsets
    executor = ProcessPoolExecutor(max_workers=8)
    futures = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(...)
        future = executor.submit(worker, shm.name, start, end, i)
        futures.append(future)

    # 5. Collect results
    results = [f.result() for f in futures]

    # 6. Cleanup
    executor.shutdown()
    shm.close()
    shm.unlink()

    return results

# ~40+ lines of boilerplate!
""")

    print("\nWITH decorator (minimal):")
    print("-" * 70)
    print("""
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze(chunk_data, chunk_id, total_chunks):
    # Just process this chunk!
    result = []
    for i in range(min(16, len(chunk_data))):
        result.append(chunk_data[i])
    return {'chunk_id': chunk_id, 'signature': result}

# Usage:
results = analyze(binary_data)

# Just ~5 lines of logic!
""")

    print()
    print("=" * 70)
    print("ANSWER: @shared_memory_task(num_chunks=N)")
    print("=" * 70)
    print()
    print("User writes:")
    print("  1. The chunk processing logic (~5 lines)")
    print()
    print("ida-taskr handles:")
    print("  1. Creating shared memory")
    print("  2. Copying data once")
    print("  3. Calculating chunk boundaries")
    print("  4. Creating workers")
    print("  5. Attaching to shared memory")
    print("  6. Cleanup (memoryview, close, unlink)")
    print()
    print("Smallest surface area: Just write the chunk logic!")
    print("=" * 70)
