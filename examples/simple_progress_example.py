"""
Simple example: CPU-intensive task without blocking

Shows the minimal code needed to run a long task in the background.
"""

from ida_taskr import ProcessPoolExecutor
import time


# Step 1: Define your worker function
def create_binary_signature(address, data, max_length=32):
    """
    Your CPU-intensive function.

    This runs in a separate process, so it won't block the main thread/UI.

    Args:
        address: Address to generate signature for
        data: Binary data bytes
        max_length: Maximum signature length

    Returns:
        Dictionary with signature results
    """
    signature = []
    total = min(max_length, len(data))

    for i in range(total):
        # Do your CPU-intensive work here
        byte_val = data[i]
        signature.append(byte_val)

        # Simulate CPU work
        time.sleep(0.01)

    # Return final result
    return {
        'address': address,
        'signature': signature,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }


# Step 2: Set up executor and connect signals
def run_signature_generation():
    """Run the task without blocking."""

    # Create executor (uses separate processes for true parallelism)
    executor = ProcessPoolExecutor(max_workers=2)

    # Connect to completion signal
    @executor.signals.on('task_completed')
    def on_completed(future):
        result = future.result()
        print(f"\n✓ Done! Signature for 0x{result['address']:X}:")
        print(f"  Pattern: {result['pattern']}")
        print(f"  Length: {len(result['signature'])} bytes")

    # Connect to error signal
    @executor.signals.on('task_failed')
    def on_failed(future, exception):
        print(f"\n✗ Error: {exception}")

    # Step 3: Submit the task (returns immediately!)
    binary_data = bytes(range(256)) * 100

    print("Submitting task to background process...")
    print("(Main thread is NOT blocked)\n")

    future = executor.submit(
        create_binary_signature,
        address=0x401000,
        data=binary_data,
        max_length=32
    )

    # The main thread continues immediately!
    # Results arrive via Qt signals when done

    return executor, future


if __name__ == '__main__':
    print("=" * 60)
    print("CPU-Intensive Task Example")
    print("=" * 60)
    print()

    executor, future = run_signature_generation()

    # Main thread is free to do other work
    print("Main thread doing other work...")
    for i in range(3):
        print(f"  Tick {i+1}/3...")
        time.sleep(1)

    print("\n✓ Main thread was never blocked!")
    print("  Worker ran in parallel in the background")
    print("=" * 60)
