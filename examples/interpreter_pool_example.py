"""
Example demonstrating InterpreterPoolExecutor usage.

This mimics the pattern from the aiointerpreters blog post:
https://github.com/Jamie-Chang/Jamie-Blog/blob/main/content/aiointerpreters.md

InterpreterPoolExecutor provides the same API as Python 3.13+'s
concurrent.futures.InterpreterPoolExecutor, but uses ProcessPoolExecutor
as the backend for compatibility with embedded Python contexts (like IDA Pro).
"""

from ida_taskr import InterpreterPoolExecutor


# Simple CPU-bound function that can be executed in parallel
def sums(num: int) -> int:
    """Compute sum of squares from 0 to num."""
    return sum(i * i for i in range(num + 1))


def main():
    """Demonstrate InterpreterPoolExecutor usage patterns."""

    # Pattern 1: Basic map() usage (from the blog post)
    print("Pattern 1: Using map()")
    with InterpreterPoolExecutor() as executor:
        results = list(executor.map(sums, [100_000] * 4))
        print(f"Results: {results}")

    # Pattern 2: Using submit() with as_completed
    print("\nPattern 2: Using submit() with as_completed")
    import concurrent.futures

    with InterpreterPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sums, 50_000) for _ in range(4)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Completed: {result}")

    # Pattern 3: With Qt signal integration
    print("\nPattern 3: With Qt signals")
    from ida_taskr import InterpreterPoolExecutor

    def on_completed(future):
        print(f"Signal: Task completed with result {future.result()}")

    def on_failed(future, exc):
        print(f"Signal: Task failed with {exc}")

    executor = InterpreterPoolExecutor(max_workers=2)
    executor.signals.task_completed.connect(on_completed)
    executor.signals.task_failed.connect(on_failed)

    future = executor.submit(sums, 10_000)
    result = future.result()  # Wait for completion
    print(f"Direct result: {result}")

    executor.shutdown(wait=True)

    print("\nAll patterns demonstrated successfully!")


if __name__ == "__main__":
    main()
