"""
Real-world IDA Pro example using SharedMemoryExecutor.

This example shows how to use SharedMemoryExecutor to analyze a binary
in IDA Pro without freezing the UI.

Use case: Finding byte patterns across a large binary segment.
"""

from ida_taskr import SharedMemoryExecutor


def find_byte_patterns(data):
    """
    Pure function that finds interesting byte patterns in data.

    This function doesn't know about chunks - it just processes data!
    """
    patterns = []

    # Look for common x86_64 function prologues
    prologue_patterns = [
        b'\x55\x48\x89\xE5',  # push rbp; mov rbp, rsp
        b'\x48\x83\xEC',      # sub rsp, X
        b'\x48\x89\x5C\x24',  # mov [rsp+X], rbx
    ]

    for pattern in prologue_patterns:
        i = 0
        while i < len(data) - len(pattern):
            if bytes(data[i:i+len(pattern)]) == pattern:
                patterns.append({
                    'offset': i,
                    'pattern': pattern.hex(),
                    'bytes': list(pattern)
                })
            i += 1

    return patterns


def example_standalone():
    """
    Standalone example (no IDA required) showing the API.
    """
    print("=" * 60)
    print("Standalone Example: Finding patterns in binary data")
    print("=" * 60)

    # Simulate binary data (8MB)
    data_size = 8 * 1024 * 1024
    binary_data = bytearray(data_size)

    # Add some fake x86_64 function prologues
    prologue = b'\x55\x48\x89\xE5'
    for i in range(0, data_size, 50000):
        if i + len(prologue) < data_size:
            binary_data[i:i+len(prologue)] = prologue

    print(f"Data size: {len(binary_data)} bytes")
    print(f"Processing in 8 chunks...\n")

    # Create executor
    executor = SharedMemoryExecutor(max_workers=8)

    # Track progress
    def on_chunk_done(chunk_id, result):
        print(f"  Chunk {chunk_id}: Found {len(result)} patterns")

    executor.signals.chunk_completed.connect(on_chunk_done)

    try:
        # Process with combining to flatten results
        future = executor.submit_chunked(
            find_byte_patterns,
            bytes(binary_data),
            num_chunks=8,
            combine=lambda results: sum(results, [])  # Flatten
        )

        # Get all patterns
        all_patterns = future.result()

        print(f"\n✅ Found {len(all_patterns)} total patterns")
        print(f"First 5 patterns:")
        for p in all_patterns[:5]:
            print(f"  Offset 0x{p['offset']:08x}: {p['pattern']}")

    finally:
        executor.shutdown(wait=True)


def example_ida_pro():
    """
    Real IDA Pro example.

    This shows how to use SharedMemoryExecutor in an IDA plugin
    to analyze the current binary without freezing the UI.
    """
    print("\n" + "=" * 60)
    print("IDA Pro Example: Analyzing current segment")
    print("=" * 60)

    try:
        import ida_bytes
        import ida_segment
        import idaapi
    except ImportError:
        print("⚠️  IDA modules not available - run this in IDA Pro")
        return

    # Get current segment
    seg = ida_segment.get_first_seg()
    if not seg:
        print("❌ No segment found")
        return

    seg_start = seg.start_ea
    seg_end = seg.end_ea
    seg_size = seg_end - seg_start

    print(f"Segment: 0x{seg_start:x} - 0x{seg_end:x}")
    print(f"Size: {seg_size} bytes ({seg_size / 1024 / 1024:.2f} MB)")

    # Get segment data
    print("\nReading segment data...")
    binary_data = ida_bytes.get_bytes(seg_start, seg_size)

    if not binary_data:
        print("❌ Failed to read segment data")
        return

    print(f"Read {len(binary_data)} bytes")

    # Show wait box (IDA UI feedback)
    idaapi.show_wait_box("HIDECANCEL\nAnalyzing patterns...")

    # Create executor
    executor = SharedMemoryExecutor(max_workers=8)

    # Track progress
    chunks_done = [0]

    def on_chunk_done(chunk_id, result):
        chunks_done[0] += 1
        progress = int((chunks_done[0] / 8) * 100)
        idaapi.replace_wait_box(f"Analyzing patterns... {progress}%")
        print(f"  Chunk {chunk_id}: Found {len(result)} patterns")

    executor.signals.chunk_completed.connect(on_chunk_done)

    try:
        # Process in chunks
        print("\nProcessing in 8 parallel chunks...")
        future = executor.submit_chunked(
            find_byte_patterns,
            binary_data,
            num_chunks=8,
            combine=lambda results: sum(results, [])
        )

        # Wait for results (IDA UI stays responsive!)
        all_patterns = future.result()

        # Hide wait box
        idaapi.hide_wait_box()

        print(f"\n✅ Found {len(all_patterns)} patterns total")

        # Convert chunk-relative offsets to absolute addresses
        for pattern in all_patterns[:10]:  # Show first 10
            abs_addr = seg_start + pattern['offset']
            print(f"  0x{abs_addr:x}: {pattern['pattern']}")

        # Could create bookmarks, comments, or colors in IDA here
        # Example:
        # for pattern in all_patterns:
        #     addr = seg_start + pattern['offset']
        #     idaapi.add_bookmark(addr, f"Pattern: {pattern['pattern']}")

    except Exception as e:
        idaapi.hide_wait_box()
        print(f"❌ Error: {e}")
        raise
    finally:
        executor.shutdown(wait=True)

    print("\n✅ Analysis complete - IDA UI never froze!")


def main():
    """
    Run both examples.
    """
    # Always run standalone example
    example_standalone()

    # Try to run IDA example if available
    try:
        import ida_bytes
        example_ida_pro()
    except ImportError:
        print("\n" + "=" * 60)
        print("IDA Pro Example: Not Available")
        print("=" * 60)
        print("⚠️  Run this script in IDA Pro to see the IDA-specific example")
        print("\nIn IDA Pro:")
        print("  1. File → Script file...")
        print("  2. Select this script")
        print("  3. IDA UI stays responsive during analysis!")


if __name__ == "__main__":
    main()
