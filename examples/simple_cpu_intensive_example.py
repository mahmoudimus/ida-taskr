"""
Simple example: CPU-intensive task without blocking IDA

This shows how to run a CPU-intensive task (like binary signature generation)
in the background without freezing IDA's UI.
"""

from ida_taskr import ProcessPoolExecutor
import time


def create_binary_signature(address, data, max_length=32):
    """
    CPU-intensive function that generates a binary signature.

    This runs in a separate process automatically, so it won't block IDA.

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
        # Your CPU-intensive work goes here
        byte_val = data[i]
        signature.append(byte_val)

        # Simulate CPU work (in real code, this would be pattern analysis)
        time.sleep(0.01)

    return {
        'address': address,
        'signature': signature,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }


def main():
    """Example of running CPU-intensive task without blocking."""

    # Create executor (uses separate processes for true parallelism)
    executor = ProcessPoolExecutor(max_workers=2)

    # Connect Qt signal to handle completion
    def on_completed(future):
        result = future.result()
        print(f"\n✓ Task completed! Signature for 0x{result['address']:X}:")
        print(f"  Pattern: {result['pattern']}")
        print(f"  Length: {len(result['signature'])} bytes")

    executor.signals.task_completed.connect(on_completed)

    # Connect Qt signal to handle errors
    def on_failed(future, exception):
        print(f"\n✗ Task failed: {exception}")

    executor.signals.task_failed.connect(on_failed)

    # Submit the task (returns immediately - doesn't block!)
    binary_data = bytes(range(256)) * 100

    print("Submitting CPU-intensive task to background process...")
    print("(IDA's UI remains responsive)\n")

    future = executor.submit(
        create_binary_signature,
        address=0x401000,
        data=binary_data,
        max_length=32
    )

    # Main thread continues immediately - not blocked!
    print("Main thread continues working...")
    for i in range(3):
        print(f"  Doing other work... {i+1}/3")
        time.sleep(1)

    # Optionally wait for the result
    result = future.result(timeout=10)

    print("\n✓ Main thread was never blocked!")
    print("  Worker ran in parallel in separate process")

    executor.shutdown(wait=True)


if __name__ == '__main__':
    print("=" * 60)
    print("CPU-Intensive Task Example (Non-Blocking)")
    print("=" * 60)
    print()

    main()

    print("\n" + "=" * 60)
