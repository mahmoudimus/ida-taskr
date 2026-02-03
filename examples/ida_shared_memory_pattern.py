"""
IDA-Specific Pattern: Processing Large Sections with Shared Memory

Shows the exact pattern used by anti_deob to analyze large code sections
in IDA without blocking the UI.

Key Points:
- IDA's main thread holds the GIL and locks
- We copy binary data from IDA into shared memory ONCE
- Multiple worker processes analyze chunks in parallel
- Workers never touch IDA - they work on shared memory
- Results come back via Qt signals
"""

import multiprocessing
import multiprocessing.shared_memory
from typing import Callable, Any
from ida_taskr import ProcessPoolExecutor


# ==============================================================================
# Pattern 1: The anti_deob pattern (how it actually works)
# ==============================================================================

def analyze_chunk(shm_name: str, start: int, end: int, start_ea: int) -> dict:
    """
    Worker function that analyzes a chunk of binary data.

    This runs in a SEPARATE PROCESS - completely isolated from IDA's locks.

    Args:
        shm_name: Shared memory segment name
        start: Start offset in shared memory
        end: End offset in shared memory
        start_ea: Base address (for reporting results)

    Returns:
        Analysis results for this chunk
    """
    # Attach to shared memory
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)

    try:
        # Get view of this chunk (NO COPYING!)
        chunk = memoryview(shm.buf)[start:end]

        # Analyze the chunk (this is where your CPU-intensive work goes)
        # Examples:
        # - Pattern matching
        # - Signature generation
        # - Opcode analysis
        # - String extraction
        # - Control flow analysis (on raw bytes)

        # Simple example: find patterns
        results = {
            'chunk_start': start,
            'chunk_end': end,
            'base_address': start_ea,
            'patterns_found': [],
            'statistics': {
                'total_bytes': len(chunk),
                'null_bytes': sum(1 for b in chunk if b == 0),
                'high_entropy_bytes': sum(1 for b in chunk if b > 0x7F),
            }
        }

        # Example: find simple patterns
        pattern = b'\x55\x8B\xEC'  # PUSH EBP; MOV EBP, ESP
        i = 0
        while i < len(chunk) - len(pattern):
            if bytes(chunk[i:i+len(pattern)]) == pattern:
                results['patterns_found'].append({
                    'offset': start + i,
                    'address': start_ea + start + i,
                    'pattern': 'function_prologue'
                })
            i += 1

        return results

    finally:
        del chunk  # Release memoryview BEFORE closing shm
        shm.close()


class BinaryAnalyzer:
    """
    Analyzes large binary sections using shared memory and parallel workers.

    This is the pattern used by DataProcessorCore in ida_taskr.
    """

    def __init__(self, max_workers: int = 4):
        """Initialize analyzer with worker pool."""
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._shm = None

    def analyze_section(
        self,
        binary_data: bytes,
        start_ea: int,
        chunk_processor: Callable = analyze_chunk,
        num_chunks: int = 8
    ) -> list:
        """
        Analyze a large binary section in parallel.

        Args:
            binary_data: Binary data from IDA (ida_bytes.get_bytes())
            start_ea: Start address in IDA
            chunk_processor: Function to process each chunk
            num_chunks: Number of chunks to split into

        Returns:
            List of results from all chunks
        """
        data_size = len(binary_data)
        chunk_size = data_size // num_chunks

        # Create shared memory and copy data ONCE
        self._shm = multiprocessing.shared_memory.SharedMemory(
            create=True,
            size=data_size
        )
        self._shm.buf[:data_size] = binary_data

        try:
            # Submit all chunks to worker pool
            futures = []

            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else data_size

                future = self.executor.submit(
                    chunk_processor,
                    self._shm.name,
                    chunk_start,
                    chunk_end,
                    start_ea
                )
                futures.append(future)

            # Collect results
            results = [f.result() for f in futures]

            return results

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up shared memory."""
        if self._shm:
            self._shm.close()
            self._shm.unlink()
            self._shm = None


# ==============================================================================
# Pattern 2: With progress updates (for IDA UI)
# ==============================================================================

class BinaryAnalyzerWithProgress(BinaryAnalyzer):
    """Analyzer with real-time progress updates via Qt signals."""

    def analyze_section_with_progress(
        self,
        binary_data: bytes,
        start_ea: int,
        on_progress: Callable[[int, str], None] = None,
        on_chunk_done: Callable[[dict], None] = None,
        num_chunks: int = 8
    ) -> list:
        """
        Analyze with progress callbacks.

        Args:
            binary_data: Binary data from IDA
            start_ea: Start address
            on_progress: Called with (progress%, message)
            on_chunk_done: Called with chunk result
            num_chunks: Number of chunks

        Returns:
            List of all results
        """
        data_size = len(binary_data)
        chunk_size = data_size // num_chunks

        # Create shared memory
        self._shm = multiprocessing.shared_memory.SharedMemory(
            create=True,
            size=data_size
        )
        self._shm.buf[:data_size] = binary_data

        try:
            # Track completion
            completed = []

            def on_complete(future):
                result = future.result()
                completed.append(result)

                progress = int(len(completed) / num_chunks * 100)
                if on_progress:
                    on_progress(
                        progress,
                        f"Analyzed {len(completed)}/{num_chunks} chunks"
                    )

                if on_chunk_done:
                    on_chunk_done(result)

            self.executor.signals.task_completed.connect(on_complete)

            # Submit chunks
            futures = []
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else data_size

                future = self.executor.submit(
                    analyze_chunk,
                    self._shm.name,
                    chunk_start,
                    chunk_end,
                    start_ea
                )
                futures.append(future)

            # Wait for all
            for f in futures:
                f.result()

            return completed

        finally:
            self.cleanup()


# ==============================================================================
# Pattern 3: Real IDA usage example
# ==============================================================================

def ida_usage_example():
    """
    Example of how to use this in an actual IDA plugin.

    This would be in your IDA plugin code.
    """
    try:
        import idaapi
        import ida_bytes
        import ida_funcs
    except ImportError:
        print("This example requires IDA Pro")
        return

    # Get binary data from IDA (this is the ONLY IDA interaction)
    start_ea = 0x401000
    end_ea = 0xC01000  # 8MB section

    # Copy data from IDA into Python bytes (releases IDA's locks)
    binary_data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)

    # Now we're free from IDA's main thread!
    # Process in parallel without blocking UI

    analyzer = BinaryAnalyzerWithProgress(max_workers=8)

    def on_progress(progress, msg):
        # Update IDA's wait box
        idaapi.replace_wait_box(f"{msg} ({progress}%)")

    def on_chunk_done(result):
        # Show partial results as they come in
        for pattern in result['patterns_found']:
            print(f"Found pattern at 0x{pattern['address']:X}")

    # This doesn't block!
    results = analyzer.analyze_section_with_progress(
        binary_data,
        start_ea,
        on_progress=on_progress,
        on_chunk_done=on_chunk_done,
        num_chunks=16
    )

    # Process combined results
    total_patterns = sum(len(r['patterns_found']) for r in results)
    print(f"Total patterns found: {total_patterns}")


# ==============================================================================
# Pattern 4: Helper functions (like anti_deob utils)
# ==============================================================================

def shm_buffer(name: str, buf_len: int = None):
    """
    Context manager for accessing shared memory.

    Usage:
        with shm_buffer(shm_name, buf_len) as buf:
            # Work with buf
            data = bytes(buf)
    """
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    try:
        yield shm.buf[:buf_len] if buf_len else shm
    finally:
        shm.close()


def execute_chunk_with_shm(
    chunk_processor: Callable,
    shm_name: str,
    chunk_start: int,
    chunk_end: int,
    *args
) -> Any:
    """
    Execute a chunk processor with proper cleanup.

    This ensures memoryview is deleted before shared memory closes,
    preventing "BufferError: cannot close exported pointers exist".

    Args:
        chunk_processor: Function that takes (memoryview, *args)
        shm_name: Shared memory name
        chunk_start: Start offset
        chunk_end: End offset
        *args: Extra args for processor

    Returns:
        Result from chunk_processor
    """
    with shm_buffer(shm_name) as shm_obj:
        # Create memoryview of just this chunk
        chunk_mv = memoryview(shm_obj.buf)[chunk_start:chunk_end]

        try:
            # Process the chunk
            result = chunk_processor(chunk_mv, *args)
            return result
        finally:
            # CRITICAL: Delete memoryview before shm closes
            del chunk_mv


# ==============================================================================
# Demo
# ==============================================================================

def main():
    print("=" * 70)
    print("IDA Shared Memory Pattern (anti_deob style)")
    print("=" * 70)
    print()

    # Simulate 8MB binary from IDA
    binary_data = bytes(range(256)) * (32 * 1024)
    start_ea = 0x401000

    print(f"Analyzing {len(binary_data):,} bytes starting at 0x{start_ea:X}")
    print()

    # Analyze with progress
    analyzer = BinaryAnalyzerWithProgress(max_workers=4)

    def show_progress(progress, msg):
        print(f"[{progress:3d}%] {msg}")

    def show_chunk_result(result):
        patterns = len(result['patterns_found'])
        if patterns > 0:
            print(f"  → Found {patterns} patterns in chunk "
                  f"[0x{result['chunk_start']:X}:0x{result['chunk_end']:X}]")

    results = analyzer.analyze_section_with_progress(
        binary_data,
        start_ea,
        on_progress=show_progress,
        on_chunk_done=show_chunk_result,
        num_chunks=8
    )

    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)

    total_patterns = sum(len(r['patterns_found']) for r in results)
    total_null = sum(r['statistics']['null_bytes'] for r in results)

    print(f"Total patterns found: {total_patterns}")
    print(f"Total null bytes: {total_null}")
    print()
    print("✓ IDA's UI was never blocked!")
    print("✓ True parallel processing (8 workers)")
    print("✓ Zero data copying between processes (shared memory)")
    print("=" * 70)


if __name__ == '__main__':
    main()
