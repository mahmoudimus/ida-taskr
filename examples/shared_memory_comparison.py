"""
Shared Memory API Comparison

Shows the difference between manual shared memory setup vs @shared_memory_task decorator.
"""

# ==============================================================================
# BEFORE: Manual Shared Memory Setup (~40+ lines)
# ==============================================================================

def analyze_manual(binary_data):
    """
    Manual shared memory - COMPLEX!

    User must handle:
    - SharedMemory creation
    - Data copying
    - Chunk boundary calculation
    - Worker function with attachment logic
    - Chunk submission
    - Result collection
    - Memoryview cleanup
    - Shared memory cleanup

    ~40+ lines of boilerplate code!
    """
    import multiprocessing.shared_memory
    from ida_taskr import ProcessPoolExecutor

    # 1. Create shared memory
    shm = multiprocessing.shared_memory.SharedMemory(
        create=True,
        size=len(binary_data)
    )

    try:
        # 2. Copy data into shared memory
        shm.buf[:len(binary_data)] = binary_data

        # 3. Calculate chunk boundaries
        num_chunks = 8
        chunk_size = len(binary_data) // num_chunks

        # 4. Define worker that attaches to shared memory
        def process_chunk(shm_name, start, end, chunk_id):
            """Worker must handle shared memory attachment."""
            shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
            try:
                # Attach to shared memory
                chunk_data = memoryview(shm.buf)[start:end]

                # Process chunk
                signatures = []
                for i in range(min(16, len(chunk_data))):
                    signatures.append(chunk_data[i])

                return {'chunk': chunk_id, 'sigs': signatures}

            finally:
                # CRITICAL: cleanup memoryview before close
                del chunk_data
                shm.close()

        # 5. Submit chunks to executor
        executor = ProcessPoolExecutor(max_workers=8)
        futures = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(binary_data)

            future = executor.submit(process_chunk, shm.name, start, end, i)
            futures.append(future)

        # 6. Collect results
        results = [f.result() for f in futures]

        # 7. Shutdown executor
        executor.shutdown()

    finally:
        # 8. Cleanup shared memory
        shm.close()
        shm.unlink()

    return results


# ==============================================================================
# AFTER: With @shared_memory_task Decorator (~5 lines!)
# ==============================================================================

from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze_decorator(chunk_data, chunk_id, total_chunks):
    """
    With decorator - SIMPLE!

    User just writes chunk processing logic.
    ida-taskr handles ALL the shared memory complexity!

    Just ~5 lines of actual logic!
    """
    # Process chunk
    signatures = []
    for i in range(min(16, len(chunk_data))):
        signatures.append(chunk_data[i])

    return {'chunk': chunk_id, 'sigs': signatures}


# Usage is also simpler:
# results = analyze_decorator(binary_data)  # That's it!


# ==============================================================================
# Real-World IDA Example
# ==============================================================================

@shared_memory_task(num_chunks=16)
def find_function_patterns(chunk_data, chunk_id, total_chunks):
    """
    Find function prologue patterns in binary data.

    Real-world use case: Analyzing large binary sections in IDA.
    """
    patterns = []

    # Common x64 function prologues
    prologue_patterns = [
        b'\x55\x48\x89\xe5',        # push rbp; mov rbp, rsp
        b'\x48\x83\xec',            # sub rsp, imm
        b'\x48\x89\x5c\x24',        # mov [rsp+X], rbx
    ]

    for i in range(len(chunk_data) - 4):
        for pattern in prologue_patterns:
            if chunk_data[i:i+len(pattern)] == pattern:
                patterns.append({
                    'chunk': chunk_id,
                    'offset': i,
                    'pattern_type': pattern.hex(),
                })

    return patterns


def ida_usage_example():
    """
    How to use this in IDA Pro.

    This is what users would write in their IDA plugins.
    """
    print("=" * 70)
    print("IDA Pro Usage Example")
    print("=" * 70)
    print()
    print("In your IDA plugin:")
    print("-" * 70)
    print("""
import ida_bytes
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=16)
def analyze_section(chunk_data, chunk_id, total_chunks):
    # Your analysis logic here
    results = find_interesting_patterns(chunk_data)
    return results

# In your plugin action:
def analyze_current_section():
    # Get binary data from IDA
    start_ea = idc.get_segm_start(idc.here())
    end_ea = idc.get_segm_end(idc.here())
    binary_data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)

    # Process in parallel - just one line!
    results = analyze_section(binary_data)

    # Show results
    for chunk_results in results:
        print(f"Chunk {chunk_results['chunk']}: {len(chunk_results)} patterns")
""")
    print()
    print("That's it! ida-taskr handles all shared memory complexity.")
    print("=" * 70)


# ==============================================================================
# Comparison Summary
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SHARED MEMORY API COMPARISON")
    print("=" * 70)
    print()

    print("WITHOUT Decorator (Manual):")
    print("-" * 70)
    print("  Lines of code: ~40-50 lines")
    print("  User handles:")
    print("    - SharedMemory creation")
    print("    - Data copying")
    print("    - Chunk boundary math")
    print("    - Worker attachment logic")
    print("    - Chunk submission")
    print("    - Result collection")
    print("    - Memoryview cleanup")
    print("    - SharedMemory cleanup")
    print()

    print("WITH Decorator (Minimal):")
    print("-" * 70)
    print("  Lines of code: ~5-10 lines")
    print("  User writes:")
    print("    - Just the chunk processing logic!")
    print()
    print("  ida-taskr handles:")
    print("    - Everything else automatically!")
    print()

    print("=" * 70)
    print("ANSWER: @shared_memory_task(num_chunks=N)")
    print("=" * 70)
    print()
    print("Smallest surface area:")
    print()
    print("  @shared_memory_task(num_chunks=8)")
    print("  def process_chunk(chunk_data, chunk_id, total_chunks):")
    print("      # Your logic here")
    print("      return result")
    print()
    print("  # Usage:")
    print("  results = process_chunk(binary_data)")
    print()
    print("That's it! User writes ~5 lines of chunk logic.")
    print("ida-taskr handles all ~40 lines of shared memory boilerplate!")
    print("=" * 70)
    print()

    # Show IDA usage
    ida_usage_example()
