"""
Ultra-Minimal Example - The Absolute Smallest Code

This is it. The smallest amount of code to use ida-taskr.
"""

from ida_taskr import cpu_task


# ==============================================================================
# The smallest amount: Just @cpu_task
# ==============================================================================

@cpu_task
def analyze(data):
    """
    Your CPU-intensive function.

    That's it. Add @cpu_task and it runs in the background.
    """
    # Your code here
    result = []
    for byte in data[:32]:
        result.append(byte)
    return result


# ==============================================================================
# Usage
# ==============================================================================

if __name__ == '__main__':
    # Create test data
    data = bytes(range(256))

    # Call it - returns immediately!
    future = analyze(data)

    # Get result when needed
    result = future.result()

    print(f"✓ Analyzed {len(result)} bytes")
    print(f"✓ Main thread never blocked")
    print()
    print("=" * 60)
    print("That's the smallest amount: @cpu_task")
    print("=" * 60)
    print()
    print("Just add one line:")
    print()
    print("    @cpu_task")
    print("    def your_function(args):")
    print("        # your code")
    print("        ...")
    print()
    print("Done!")
