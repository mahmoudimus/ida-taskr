"""
Comparison: @shared_memory_task decorator vs SharedMemoryExecutor

This example shows the two approaches for shared memory processing:
1. Decorator approach (@shared_memory_task) - Simplest, minimal code
2. Executor approach (SharedMemoryExecutor) - More control, standard interface

Both use the same underlying shared memory mechanism.
"""

from ida_taskr import shared_memory_task, SharedMemoryExecutor


# ============================================================================
# Approach 1: Decorator (Simplest)
# ============================================================================
@shared_memory_task(num_chunks=8)
def analyze_with_decorator(chunk_data, chunk_id, total_chunks):
    """
    Decorator approach: Function receives chunk-specific parameters.

    Pros:
    - Simplest API (just add decorator)
    - Automatic executor management
    - Good for quick scripts

    Cons:
    - Function signature includes chunk parameters
    - Less control over executor lifecycle
    """
    # Find patterns in this chunk
    count = sum(1 for byte in chunk_data if byte > 0x80)
    return {'chunk_id': chunk_id, 'high_bytes': count}


# ============================================================================
# Approach 2: Executor (More control)
# ============================================================================
def analyze_pure(data):
    """
    Executor approach: Pure function, doesn't know about chunks.

    Pros:
    - Pure function signature (reusable)
    - Full control over executor lifecycle
    - Standard concurrent.futures interface
    - Can use same executor for multiple operations
    - Qt signal support for progress tracking

    Cons:
    - Slightly more verbose (create executor, call submit_chunked)
    """
    # Same logic, but function doesn't know about chunks
    count = sum(1 for byte in data if byte > 0x80)
    return count


def main():
    # Create test data
    print("Creating test data...")
    data_size = 8 * 1024 * 1024
    binary_data = bytearray(data_size)

    # Fill with some high bytes
    for i in range(0, data_size, 100):
        binary_data[i] = 0xFF

    print(f"Data size: {len(binary_data)} bytes\n")

    # ========================================================================
    # Approach 1: Decorator
    # ========================================================================
    print("=" * 60)
    print("Approach 1: @shared_memory_task decorator")
    print("=" * 60)

    # Just call the decorated function with full data
    results = analyze_with_decorator(bytes(binary_data))

    print(f"Processed {len(results)} chunks")
    total_high_bytes = sum(r['high_bytes'] for r in results)
    print(f"Total high bytes found: {total_high_bytes}")

    # ========================================================================
    # Approach 2: Executor
    # ========================================================================
    print("\n" + "=" * 60)
    print("Approach 2: SharedMemoryExecutor")
    print("=" * 60)

    # Create executor (can reuse for multiple operations)
    executor = SharedMemoryExecutor(max_workers=8)

    try:
        # Option 2a: Get list of chunk results
        print("\nOption 2a: submit_chunked (list of results)")
        future = executor.submit_chunked(analyze_pure, bytes(binary_data), num_chunks=8)
        chunk_results = future.result()
        print(f"Chunk results: {chunk_results}")

        # Option 2b: Combine results automatically
        print("\nOption 2b: submit_chunked with combine")
        future = executor.submit_chunked(
            analyze_pure,
            bytes(binary_data),
            num_chunks=8,
            combine=sum  # Sum all chunk counts
        )
        total = future.result()
        print(f"Total high bytes: {total}")

        # Option 2c: Stream results
        print("\nOption 2c: map_chunked (streaming)")
        total_streaming = 0
        for i, count in enumerate(executor.map_chunked(analyze_pure, bytes(binary_data), num_chunks=8)):
            print(f"  Chunk {i}: {count} high bytes")
            total_streaming += count
        print(f"Total: {total_streaming}")

        # Bonus: Can use standard interface too
        print("\nBonus: Standard submit() interface also available")
        future = executor.submit(lambda x: x * 2, 42)
        print(f"Standard submit result: {future.result()}")

    finally:
        executor.shutdown(wait=True)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
When to use each:

✅ Use @shared_memory_task when:
   - You want the simplest API
   - Writing quick scripts or plugins
   - Don't need executor control

✅ Use SharedMemoryExecutor when:
   - You want pure function signatures
   - Need control over executor lifecycle
   - Want to reuse executor for multiple operations
   - Need Qt signal integration for progress tracking
   - Want standard concurrent.futures interface
   - Building libraries or complex applications

Both approaches:
   - Use same shared memory mechanism
   - Zero-copy for workers
   - Automatic cleanup
   - 8x faster than copying for large data
""")


if __name__ == "__main__":
    main()
