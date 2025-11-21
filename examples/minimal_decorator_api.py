"""
Minimal Decorator API - The Smallest Amount of Code Possible

Shows the absolute simplest way to use ida-taskr decorators.
"""

import time
from ida_taskr import cpu_task


# ==============================================================================
# BEFORE: Without decorator (manual setup - lots of boilerplate)
# ==============================================================================

def analyze_binary_manual(data):
    """Old way: Manual ProcessPoolExecutor setup."""
    from ida_taskr import ProcessPoolExecutor

    def worker(data):
        # Do CPU work
        result = []
        for i in range(min(16, len(data))):
            result.append(data[i])
            time.sleep(0.01)
        return result

    # Manual setup (boilerplate)
    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker, data)
    result = future.result()
    executor.shutdown()

    return result


# ==============================================================================
# AFTER: With minimal decorator (ONE LINE!)
# ==============================================================================

@cpu_task
def analyze_binary(data):
    """
    New way: Just add @cpu_task

    That's it. One line. Done.
    """
    result = []
    for i in range(min(16, len(data))):
        result.append(data[i])
        time.sleep(0.01)
    return result


# ==============================================================================
# Usage Comparison
# ==============================================================================

def demo():
    data = bytes(range(256)) * 100

    print("=" * 70)
    print("Minimal Decorator API")
    print("=" * 70)
    print()

    # Without decorator: ~8 lines of boilerplate
    print("WITHOUT decorator (manual):")
    print("-" * 70)
    print("""
def analyze_binary_manual(data):
    from ida_taskr import ProcessPoolExecutor

    def worker(data):
        # Your code here
        ...

    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker, data)
    result = future.result()
    executor.shutdown()
    return result

# 8+ lines of boilerplate!
""")

    # With decorator: 1 line!
    print("\nWITH decorator (minimal):")
    print("-" * 70)
    print("""
@cpu_task
def analyze_binary(data):
    # Your code here
    ...

# Just 1 line!
""")

    print("\n" + "=" * 70)
    print("That's the smallest amount: @cpu_task")
    print("=" * 70)
    print()

    # Show it actually works
    print("Running with decorator:")
    future = analyze_binary(data)
    print(f"  → Returns immediately: {type(future)}")
    print(f"  → Main thread free!")

    result = future.result(timeout=5)
    print(f"  → Got result: {len(result)} bytes")
    print()


if __name__ == '__main__':
    demo()
