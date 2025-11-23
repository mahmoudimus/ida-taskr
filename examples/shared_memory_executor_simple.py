"""
Simple example demonstrating SharedMemoryExecutor with submit_chunked().

This example shows how to use SharedMemoryExecutor to process large data
in chunks using shared memory with zero-copy for workers.

Key point: User function stays pure - fn(data) -> result
No need to know about chunks, shared memory, or chunk IDs!
"""

from ida_taskr import SharedMemoryExecutor


def find_patterns(data):
    """
    Pure function that finds patterns in data.

    Doesn't know about chunks or shared memory - just processes data!
    """
    # Find all positions where we see 0xFF
    positions = []
    for i in range(len(data)):
        if data[i] == 0xFF:
            positions.append(i)
    return positions


def main():
    # Create some test data (8MB)
    print("Creating 8MB test data...")
    data_size = 8 * 1024 * 1024
    binary_data = bytearray(data_size)

    # Add some 0xFF bytes at specific positions
    for i in range(0, data_size, 10000):
        binary_data[i] = 0xFF

    print(f"Data size: {len(binary_data)} bytes")

    # Create executor
    executor = SharedMemoryExecutor(max_workers=8)

    try:
        print("\n--- Example 1: submit_chunked (returns list of chunk results) ---")

        # Process data in 8 chunks using shared memory
        future = executor.submit_chunked(find_patterns, bytes(binary_data), num_chunks=8)

        # Get results - list of results from each chunk
        chunk_results = future.result()

        print(f"Processed {len(chunk_results)} chunks")
        for i, positions in enumerate(chunk_results):
            print(f"  Chunk {i}: Found {len(positions)} patterns")

        print("\n--- Example 2: submit_chunked with combine ---")

        # Same function, but flatten results automatically
        future = executor.submit_chunked(
            find_patterns,
            bytes(binary_data),
            num_chunks=8,
            combine=lambda results: sum(results, [])  # Flatten list of lists
        )

        # Get flattened results
        all_positions = future.result()
        print(f"Total patterns found: {len(all_positions)}")
        print(f"First 10 positions: {all_positions[:10]}")

        print("\n--- Example 3: map_chunked (streaming results) ---")

        # Stream results as chunks complete
        for i, positions in enumerate(executor.map_chunked(find_patterns, bytes(binary_data), num_chunks=8)):
            print(f"  Chunk completed: Found {len(positions)} patterns")

    finally:
        executor.shutdown(wait=True)

    print("\n✅ Done! Zero data copying to workers.")


if __name__ == "__main__":
    main()
