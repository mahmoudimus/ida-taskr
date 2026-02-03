"""
Super simple example using @cpu_task decorator

Shows how decorators make the API extremely simple.
"""

from ida_taskr import cpu_task
import time


# Just add @cpu_task decorator to any function!
@cpu_task(on_complete=lambda result: print(f"\n✓ Done! Pattern: {result['pattern']}"))
def generate_signature(address, data, max_length=32):
    """
    Your CPU-intensive function - just write it normally!

    The @cpu_task decorator automatically:
    - Runs it in a separate process
    - Handles all Qt signal connections
    - Calls your callback when done
    """
    signature = []

    for i in range(min(max_length, len(data))):
        signature.append(data[i])
        time.sleep(0.01)  # Simulate CPU work

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }


def main():
    """Example usage."""
    print("=" * 60)
    print("Super Simple Decorator Example")
    print("=" * 60)
    print()

    # Just call it like a normal function!
    binary_data = bytes(range(256)) * 100

    print("Calling generate_signature(0x401000, data)...")
    print("(Returns immediately, runs in background)\n")

    # This returns a Future immediately - doesn't block!
    future = generate_signature(0x401000, binary_data)

    # Main thread keeps running
    print("Main thread is free:")
    for i in range(3):
        print(f"  Working... {i+1}/3")
        time.sleep(1)

    # Optional: wait for result
    result = future.result(timeout=10)

    print("\n✓ Main thread was NEVER blocked!")
    print("=" * 60)


if __name__ == '__main__':
    main()
