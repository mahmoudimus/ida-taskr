"""
Example: Creating binary signatures without blocking IDA's UI

This example shows how to use ida-taskr to perform CPU-intensive signature
generation while keeping IDA responsive and showing progress updates.
"""

import time
from ida_taskr import TaskRunner, create_worker


# ==============================================================================
# APPROACH 1: Using TaskRunner (High-level, recommended for IDA plugins)
# ==============================================================================

def generate_signature_worker(address, data, max_length=32):
    """
    Worker function that generates a unique binary signature.

    This runs in a separate process, so it won't block IDA's UI.

    Args:
        address: Address to generate signature for
        data: Binary data to search (bytes)
        max_length: Maximum signature length

    Yields:
        Progress updates as the signature is being generated

    Returns:
        The final signature as a dict
    """
    # Simulate signature generation with progress updates
    signature = []
    total_bytes = min(max_length, len(data))

    for i in range(total_bytes):
        # Check if this byte is unique enough
        byte_val = data[i]

        # Simulate analysis (in real code, this would check uniqueness)
        time.sleep(0.01)  # Simulating CPU work

        # Emit progress update (these become Qt signals)
        progress = int((i + 1) / total_bytes * 100)
        yield {
            'type': 'progress',
            'progress': progress,
            'message': f'Analyzing byte {i+1}/{total_bytes}: 0x{byte_val:02X}'
        }

        signature.append(byte_val)

        # Check if signature is unique enough (simplified)
        if len(signature) >= 8 and i % 4 == 0:
            yield {
                'type': 'status',
                'message': f'Testing signature uniqueness at {len(signature)} bytes...'
            }

    # Return final result
    return {
        'address': address,
        'signature': signature,
        'length': len(signature),
        'pattern': ' '.join(f'{b:02X}' for b in signature)
    }


def example_taskrunner():
    """Example using TaskRunner - best for IDA plugins."""

    # Read some binary data (in IDA, you'd use ida_bytes.get_bytes())
    # For demo purposes, we'll use dummy data
    binary_data = bytes(range(256)) * 100  # 25KB of data

    # Create a TaskRunner
    runner = TaskRunner()

    # Connect to signals to receive updates
    @runner.message_emitter.on('progress')
    def on_progress(data):
        """Called when worker reports progress."""
        print(f"[Progress {data['progress']}%] {data['message']}")

    @runner.message_emitter.on('status')
    def on_status(data):
        """Called when worker reports status."""
        print(f"[Status] {data['message']}")

    @runner.message_emitter.on('result')
    def on_result(data):
        """Called when worker completes."""
        result = data['result']
        print(f"\n✓ Signature generated!")
        print(f"  Address: 0x{result['address']:X}")
        print(f"  Length: {result['length']} bytes")
        print(f"  Pattern: {result['pattern']}")

    @runner.message_emitter.on('error')
    def on_error(data):
        """Called if worker fails."""
        print(f"✗ Error: {data['error']}")

    # Start the worker (non-blocking!)
    print("Starting signature generation (UI remains responsive)...\n")
    runner.run_task(
        generate_signature_worker,
        address=0x401000,
        data=binary_data,
        max_length=32
    )

    # IDA's UI is NOT blocked - user can continue working
    # Progress updates will appear as they happen

    return runner


# ==============================================================================
# APPROACH 2: Using ThreadExecutor (Lower-level, more control)
# ==============================================================================

def generate_signature_simple(address, data, max_length=32):
    """
    Simpler version without yield - just returns result.
    Use this with ThreadExecutor or ProcessPoolExecutor.
    """
    signature = []
    total_bytes = min(max_length, len(data))

    for i in range(total_bytes):
        byte_val = data[i]
        time.sleep(0.01)  # Simulating work
        signature.append(byte_val)

    return {
        'address': address,
        'signature': signature,
        'length': len(signature),
        'pattern': ' '.join(f'{b:02X}' for b in signature)
    }


def example_thread_executor():
    """Example using ThreadExecutor - good for I/O-bound tasks."""
    from ida_taskr import ThreadExecutor

    binary_data = bytes(range(256)) * 100

    # Create executor
    executor = ThreadExecutor(max_workers=4)

    # Connect to signals
    @executor.signals.on('task_completed')
    def on_completed(future):
        """Called when task completes."""
        result = future.result()
        print(f"\n✓ Signature generated!")
        print(f"  Pattern: {result['pattern']}")

    @executor.signals.on('task_failed')
    def on_failed(future, exception):
        """Called if task fails."""
        print(f"✗ Error: {exception}")

    # Submit task (non-blocking!)
    print("Submitting signature generation task...\n")
    future = executor.submit(
        generate_signature_simple,
        address=0x401000,
        data=binary_data,
        max_length=32
    )

    # Can submit multiple tasks in parallel!
    # future2 = executor.submit(generate_signature_simple, 0x402000, data2)
    # future3 = executor.submit(generate_signature_simple, 0x403000, data3)

    return executor


# ==============================================================================
# APPROACH 3: Using ProcessPoolExecutor (Best for CPU-intensive)
# ==============================================================================

def example_process_pool():
    """
    Example using ProcessPoolExecutor - best for CPU-intensive tasks.

    This uses separate Python processes, so it bypasses the GIL and
    can use multiple CPU cores effectively.
    """
    from ida_taskr import ProcessPoolExecutor

    binary_data = bytes(range(256)) * 100

    # Create executor with multiple workers
    executor = ProcessPoolExecutor(max_workers=4)

    # Connect to signals
    @executor.signals.on('task_completed')
    def on_completed(future):
        result = future.result()
        print(f"✓ Signature for 0x{result['address']:X}: {result['pattern'][:40]}...")

    # Submit multiple signature generation tasks in parallel
    print("Generating signatures for multiple addresses in parallel...\n")

    addresses = [0x401000, 0x402000, 0x403000, 0x404000]
    futures = []

    for addr in addresses:
        # Each runs in a separate process - true parallelism!
        future = executor.submit(
            generate_signature_simple,
            address=addr,
            data=binary_data[addr % 1000:],
            max_length=16
        )
        futures.append(future)

    print(f"Submitted {len(futures)} tasks - all running in parallel!\n")

    return executor, futures


# ==============================================================================
# REAL IDA EXAMPLE
# ==============================================================================

def ida_signature_scanner_example():
    """
    Real-world IDA Pro example: Scan function for unique signatures.

    This would be used in an IDA plugin to find signatures without
    freezing the UI.
    """
    try:
        import idaapi
        import ida_bytes
        import ida_funcs
    except ImportError:
        print("This example requires IDA Pro")
        return None

    def scan_function_for_signature(func_ea):
        """Worker that analyzes a function and finds unique signature."""
        func = ida_funcs.get_func(func_ea)
        if not func:
            return None

        # Read function bytes
        func_size = func.end_ea - func.start_ea
        func_bytes = ida_bytes.get_bytes(func.start_ea, func_size)

        # Generate signature with progress updates
        signature = []
        chunk_size = max(1, func_size // 100)  # For progress updates

        for i in range(min(64, func_size)):  # Max 64 byte signature
            if i % chunk_size == 0:
                progress = int(i / min(64, func_size) * 100)
                yield {
                    'type': 'progress',
                    'progress': progress,
                    'message': f'Scanning function at 0x{func_ea:X}...'
                }

            signature.append(func_bytes[i])

            # Check uniqueness (simplified - real code would search database)
            if len(signature) >= 16:
                break

        return {
            'function': func_ea,
            'signature': signature,
            'pattern': ' '.join(f'{b:02X}' for b in signature)
        }

    # Create TaskRunner
    runner = TaskRunner()

    # Setup UI callbacks
    @runner.message_emitter.on('progress')
    def update_ui(data):
        # Update IDA's status bar or progress dialog
        idaapi.msg(f"{data['message']}\n")

    @runner.message_emitter.on('result')
    def show_result(data):
        result = data['result']
        idaapi.msg(f"✓ Signature: {result['pattern']}\n")

    # Get current function
    func_ea = idaapi.get_screen_ea()

    # Run analysis without blocking IDA
    runner.run_task(scan_function_for_signature, func_ea)

    return runner


if __name__ == '__main__':
    print("=" * 70)
    print("ida-taskr: Non-blocking CPU-Intensive Task Examples")
    print("=" * 70)
    print()

    # Example 1: TaskRunner (recommended)
    print("EXAMPLE 1: TaskRunner (High-level API)")
    print("-" * 70)
    runner = example_taskrunner()
    print()

    # Give it time to show some progress
    time.sleep(2)

    print("\nEXAMPLE 2: ThreadExecutor")
    print("-" * 70)
    executor = example_thread_executor()
    print()

    time.sleep(2)

    print("\nEXAMPLE 3: ProcessPoolExecutor (parallel)")
    print("-" * 70)
    pool, futures = example_process_pool()
    print()

    # Keep running to see results
    time.sleep(3)

    print("\n" + "=" * 70)
    print("All tasks running in background - UI would remain responsive!")
    print("=" * 70)
