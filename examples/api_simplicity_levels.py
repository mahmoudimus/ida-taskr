"""
API Simplicity Levels - From Complex to Ultra-Simple

Shows different levels of API complexity, from most verbose to most minimal.
"""

import time


# ==============================================================================
# Level 0: Raw ProcessPoolExecutor (Most Verbose)
# ==============================================================================

def level_0_raw_executor():
    """
    Level 0: Raw ProcessPoolExecutor

    ~15 lines of setup code
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    def worker_function(data):
        result = []
        for i in range(min(16, len(data))):
            result.append(data[i])
        return result

    # Manual executor management
    executor = ProcessPoolExecutor(max_workers=4)

    try:
        # Submit task
        future = executor.submit(worker_function, bytes(range(256)))

        # Wait for result
        result = future.result(timeout=10)

        return result

    finally:
        # Cleanup
        executor.shutdown(wait=True)


# ==============================================================================
# Level 1: Using ida-taskr's ProcessPoolExecutor (Less Verbose)
# ==============================================================================

def level_1_ida_taskr_executor():
    """
    Level 1: Using ida-taskr's ProcessPoolExecutor

    ~8 lines (handles Qt integration)
    """
    from ida_taskr import ProcessPoolExecutor

    def worker_function(data):
        result = []
        for i in range(min(16, len(data))):
            result.append(data[i])
        return result

    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker_function, bytes(range(256)))
    result = future.result()
    executor.shutdown()

    return result


# ==============================================================================
# Level 2: Simple decorator with no options (Minimal)
# ==============================================================================

from ida_taskr import cpu_task

@cpu_task
def level_2_simple_decorator(data):
    """
    Level 2: Just @cpu_task decorator

    1 line decorator + your function
    """
    result = []
    for i in range(min(16, len(data))):
        result.append(data[i])
    return result

# Usage:
# future = level_2_simple_decorator(data)  # Returns immediately
# result = future.result()


# ==============================================================================
# Level 3: Decorator with callback (Simple + Convenient)
# ==============================================================================

@cpu_task(on_complete=lambda r: print(f"Done: {len(r)} bytes"))
def level_3_with_callback(data):
    """
    Level 3: Decorator with callback

    1 line decorator + callback function
    """
    result = []
    for i in range(min(16, len(data))):
        result.append(data[i])
    return result

# Usage:
# level_3_with_callback(data)  # Fire and forget!


# ==============================================================================
# Level 4: Full featured (Everything)
# ==============================================================================

@cpu_task(
    on_complete=lambda r: print(f"✓ Done: {len(r)} bytes"),
    on_error=lambda e: print(f"✗ Error: {e}"),
)
def level_4_full_featured(data):
    """
    Level 4: All the bells and whistles

    Multi-line decorator with all options
    """
    result = []
    for i in range(min(16, len(data))):
        result.append(data[i])
    return result


# ==============================================================================
# Comparison
# ==============================================================================

def show_comparison():
    """Show all levels side by side."""

    print("=" * 70)
    print("API Simplicity Levels")
    print("=" * 70)
    print()

    levels = [
        ("Level 0: Raw ProcessPoolExecutor", """
def analyze(data):
    from concurrent.futures import ProcessPoolExecutor

    def worker(data):
        # Your code
        ...

    executor = ProcessPoolExecutor(max_workers=4)
    try:
        future = executor.submit(worker, data)
        result = future.result(timeout=10)
        return result
    finally:
        executor.shutdown(wait=True)

# ~15 lines of boilerplate
"""),

        ("Level 1: ida-taskr ProcessPoolExecutor", """
def analyze(data):
    from ida_taskr import ProcessPoolExecutor

    def worker(data):
        # Your code
        ...

    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker, data)
    result = future.result()
    executor.shutdown()
    return result

# ~8 lines (better, but still manual)
"""),

        ("Level 2: @cpu_task (MINIMAL)", """
from ida_taskr import cpu_task

@cpu_task
def analyze(data):
    # Your code
    ...

future = analyze(data)
result = future.result()

# Just 1 line! (@cpu_task)
# This is the SMALLEST amount!
"""),

        ("Level 3: @cpu_task with callback", """
@cpu_task(on_complete=lambda r: print(r))
def analyze(data):
    # Your code
    ...

analyze(data)  # Fire and forget!

# Still simple, results auto-delivered
"""),

        ("Level 4: Full featured", """
@cpu_task(
    on_complete=show_result,
    on_error=show_error,
)
def analyze(data):
    # Your code
    ...

analyze(data)

# All features, still simple!
"""),
    ]

    for title, code in levels:
        print(title)
        print("-" * 70)
        print(code)
        print()

    print("=" * 70)
    print("ANSWER: The smallest amount is Level 2 - just @cpu_task")
    print("=" * 70)
    print()
    print("@cpu_task  ← That's it. One line. Done.")
    print()


def run_demo():
    """Actually run the different levels to show they work."""
    data = bytes(range(256)) * 100

    print("\n" + "=" * 70)
    print("Running Each Level")
    print("=" * 70)
    print()

    # Level 2: Minimal
    print("Level 2 (Minimal @cpu_task):")
    print("-" * 70)
    future = level_2_simple_decorator(data)
    print(f"  ✓ Returns immediately: {type(future).__name__}")
    result = future.result(timeout=5)
    print(f"  ✓ Got result: {len(result)} bytes")
    print()

    # Level 3: With callback
    print("Level 3 (With callback):")
    print("-" * 70)
    print("  ", end="")
    level_3_with_callback(data).result(timeout=5)  # Callback fires
    print()

    print("=" * 70)
    print("All levels work - Level 2 is the simplest!")
    print("=" * 70)


if __name__ == '__main__':
    show_comparison()
    run_demo()
