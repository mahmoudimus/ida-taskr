"""
Working Example: @shared_memory_task Decorator

Shows the minimal API for processing large binary data with shared memory.
"""

import time
from ida_taskr import shared_memory_task


# ==============================================================================
# MINIMAL API: Just write the chunk processing logic!
# ==============================================================================

@shared_memory_task(num_chunks=8)
def find_signatures(chunk_data, chunk_id, total_chunks):
    """
    Find binary signatures in this chunk.

    This is ALL the user needs to write!
    ida-taskr handles all the shared memory complexity.

    Args:
        chunk_data: memoryview of this chunk's data (no copying!)
        chunk_id: 0-based chunk index (0 to 7 in this case)
        total_chunks: total number of chunks (8 in this case)

    Returns:
        List of signatures found in this chunk
    """
    signatures = []

    # Process this chunk
    for i in range(len(chunk_data) - 16):
        # Simple pattern: find sequences that start with 0x48
        if chunk_data[i] == 0x48:
            # Extract 16-byte signature
            sig = bytes(chunk_data[i:i+16])
            signatures.append({
                'chunk': chunk_id,
                'offset': i,
                'signature': sig.hex()
            })

    return signatures


# ==============================================================================
# Usage - Just pass the full data!
# ==============================================================================

def demo():
    """Demonstrate the minimal shared memory API."""

    # Create test data (simulating 8MB binary)
    print("Creating test binary data (8MB)...")
    binary_data = bytearray()
    for i in range(8 * 1024 * 1024):
        binary_data.append(i % 256)

    # Add some patterns
    for offset in [0x1000, 0x2000, 0x100000, 0x500000]:
        binary_data[offset] = 0x48  # Pattern marker
        binary_data[offset + 1:offset + 4] = b'\x89\xe5\x48'

    binary_data = bytes(binary_data)

    print(f"Processing {len(binary_data):,} bytes...")
    print()

    # This is ALL the user needs to write:
    start = time.time()
    results = find_signatures(binary_data)
    elapsed = time.time() - start

    # Show results
    print(f"✓ Processed in {elapsed:.2f}s")
    print(f"✓ Found {sum(len(r) for r in results)} signatures across {len(results)} chunks")
    print()

    # Show signatures from each chunk
    for chunk_results in results:
        if chunk_results:
            chunk_id = chunk_results[0]['chunk']
            print(f"  Chunk {chunk_id}: {len(chunk_results)} signatures")
            if chunk_results:
                print(f"    First: {chunk_results[0]['signature'][:32]}...")

    print()
    print("=" * 70)
    print("That's the minimal API!")
    print("=" * 70)
    print()
    print("User wrote:")
    print("  1. The chunk processing logic (~10 lines)")
    print("  2. Called the decorated function with full data")
    print()
    print("ida-taskr handled:")
    print("  1. Creating shared memory")
    print("  2. Copying data once")
    print("  3. Calculating chunk boundaries")
    print("  4. Creating 8 worker processes")
    print("  5. Attaching workers to shared memory")
    print("  6. Collecting results")
    print("  7. Cleanup (memoryview, close, unlink)")
    print()
    print("Just @shared_memory_task(num_chunks=8)!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
