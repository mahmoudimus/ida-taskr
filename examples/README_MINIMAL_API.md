# The Smallest Amount: Minimal API Guide

## Question: What's the smallest amount of annotations and decorators to use ida-taskr?

**Answer: Just `@cpu_task`**

That's it. One line.

---

## Comparison

### Without Decorator (Manual Setup)

```python
from ida_taskr import ProcessPoolExecutor

def analyze_binary(data):
    """Old way - requires manual executor setup."""

    def worker(data):
        # Your CPU-intensive code
        result = []
        for byte in data[:32]:
            result.append(byte)
        return result

    # Manual boilerplate (10+ lines)
    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker, data)
    result = future.result()
    executor.shutdown()

    return result

# Result: ~10 lines of boilerplate code
```

### With Decorator (MINIMAL)

```python
from ida_taskr import cpu_task

@cpu_task
def analyze_binary(data):
    """New way - just add @cpu_task"""

    # Your CPU-intensive code
    result = []
    for byte in data[:32]:
        result.append(byte)
    return result

# Result: Just 1 line! (@cpu_task)
```

---

## Usage

### Basic (Fire and Forget)

```python
from ida_taskr import cpu_task

@cpu_task
def analyze(data):
    # Your code
    return result

# Call it - returns immediately!
future = analyze(data)

# Get result when needed
result = future.result()
```

### With Callback (Even Simpler)

```python
@cpu_task(on_complete=lambda r: print(f"Done: {r}"))
def analyze(data):
    # Your code
    return result

# Just call it - callback fires automatically!
analyze(data)
```

### With Error Handling

```python
@cpu_task(
    on_complete=lambda r: print(f"✓ {r}"),
    on_error=lambda e: print(f"✗ {e}")
)
def analyze(data):
    # Your code
    return result

analyze(data)
```

---

## Real-World Example: IDA Binary Analysis

### Without Decorator

```python
def find_signatures(start_ea, end_ea):
    """Find signatures in binary section - old way."""
    import ida_bytes
    from ida_taskr import ProcessPoolExecutor

    # Get data from IDA
    data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)

    # Worker function
    def worker(data):
        signatures = []
        # ... complex analysis ...
        return signatures

    # Manual executor management
    executor = ProcessPoolExecutor(max_workers=4)
    future = executor.submit(worker, data)
    result = future.result()
    executor.shutdown()

    return result

# ~15 lines of code
```

### With Decorator (MINIMAL)

```python
from ida_taskr import cpu_task

@cpu_task
def find_signatures_worker(data):
    """Find signatures - worker function."""
    signatures = []
    # ... complex analysis ...
    return signatures

def find_signatures(start_ea, end_ea):
    """Find signatures in binary section - new way."""
    import ida_bytes

    # Get data from IDA
    data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)

    # Run in background - one line!
    future = find_signatures_worker(data)

    return future.result()

# Just 1 decorator line!
```

---

## Summary: API Levels

| Level | Code | Lines |
|-------|------|-------|
| **Manual** | `ProcessPoolExecutor(...)` | ~10-15 lines |
| **Minimal** | `@cpu_task` | **1 line** ← This is it! |
| **With Callback** | `@cpu_task(on_complete=...)` | 1 line + callback |
| **Full Featured** | `@cpu_task(on_complete=..., on_error=...)` | 2-3 lines |

---

## The Answer

**The smallest amount of annotations and decorators:**

```python
@cpu_task
```

That's it. One line. Done.

---

## Examples in This Directory

- **`ultra_minimal.py`** - The absolute simplest example (just `@cpu_task`)
- **`minimal_decorator_api.py`** - Before/after comparison
- **`api_simplicity_levels.py`** - All levels from verbose to minimal
- **`decorator_simple_example.py`** - Simple real-world example
- **`decorator_evolution.py`** - Step-by-step evolution from blocking to async

Start with `ultra_minimal.py` to see the smallest possible usage.
