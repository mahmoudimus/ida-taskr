"""
ONE LINE SOLUTION

The smallest amount of code to make a function run in the background.
"""

# ==============================================================================
# BEFORE: Blocks the main thread
# ==============================================================================

def analyze_before(data):
    # Your CPU-intensive work
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)  # Expensive computation
    return result

# Problem: Calling this blocks the main thread!
# result = analyze_before(data)  ← UI freezes during execution


# ==============================================================================
# AFTER: Add @cpu_task (ONE LINE!)
# ==============================================================================

from ida_taskr import cpu_task

@cpu_task  # ← That's it. Just add this line.
def analyze_after(data):
    # Your CPU-intensive work (SAME CODE!)
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)  # Expensive computation
    return result

# Solution: Calling this returns immediately!
# future = analyze_after(data)  ← UI stays responsive
# result = future.result()      ← Get result when ready


# ==============================================================================
# Summary
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ONE LINE SOLUTION")
    print("=" * 60)
    print()
    print("Problem: Function blocks the main thread")
    print()
    print("  def analyze(data):")
    print("      # expensive work...")
    print()
    print("Solution: Add @cpu_task (ONE LINE!)")
    print()
    print("  @cpu_task  ← Just add this")
    print("  def analyze(data):")
    print("      # expensive work...")
    print()
    print("That's it!")
    print("=" * 60)
    print()
    print("The function now runs in the background.")
    print("Main thread stays responsive.")
    print("No blocking. No freezing. No complexity.")
    print()
    print("Just one line: @cpu_task")
    print("=" * 60)
