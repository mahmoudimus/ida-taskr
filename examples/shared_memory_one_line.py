"""
ONE LINE SHARED MEMORY SOLUTION

The smallest amount of code for parallel processing with shared memory.
"""

from ida_taskr import shared_memory_task


# ==============================================================================
# The Smallest Surface Area: Just @shared_memory_task(num_chunks=N)
# ==============================================================================

@shared_memory_task(num_chunks=8)
def analyze(chunk_data, chunk_id, total_chunks):
    """
    Process one chunk of data.

    That's it! User just writes the chunk logic.
    ida-taskr handles ALL shared memory complexity:
    - Creating shared memory
    - Copying data once
    - Calculating boundaries
    - Creating workers
    - Attaching to shared memory
    - Collecting results
    - Cleanup
    """
    # Your chunk processing logic
    result = []
    for i in range(min(32, len(chunk_data))):
        result.append(chunk_data[i])
    return result


# Usage - just pass the full data!
# results = analyze(binary_data)  # Returns list of all chunk results


# ==============================================================================
# Summary
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ONE LINE SHARED MEMORY SOLUTION")
    print("=" * 70)
    print()
    print("Question: What's the smallest surface area for shared memory?")
    print()
    print("Answer: Just write the chunk processing logic!")
    print()
    print("-" * 70)
    print()
    print("  @shared_memory_task(num_chunks=8)")
    print("  def analyze(chunk_data, chunk_id, total_chunks):")
    print("      # Your chunk logic here")
    print("      return result")
    print()
    print("  # Usage:")
    print("  results = analyze(binary_data)")
    print()
    print("-" * 70)
    print()
    print("What user provides:")
    print("  - Chunk processing function (~5 lines)")
    print()
    print("What ida-taskr handles:")
    print("  - SharedMemory creation")
    print("  - Copying data once into shared memory")
    print("  - Calculating chunk boundaries")
    print("  - Creating worker processes")
    print("  - Attaching workers to shared memory")
    print("  - Submitting chunks")
    print("  - Collecting results")
    print("  - Memoryview cleanup")
    print("  - SharedMemory cleanup")
    print("  (~40 lines of boilerplate!)")
    print()
    print("=" * 70)
    print()
    print("Smallest surface area: Just the chunk logic!")
    print()
    print("=" * 70)
