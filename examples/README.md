# ida-taskr Examples

Examples showing how to use ida-taskr for non-blocking, CPU-intensive tasks in IDA Pro.

## Quick Start

**Problem:** You need to do something CPU-intensive (like generating binary signatures, analyzing patterns, or searching large datasets) without freezing IDA's UI.

**Solution:** Use `TaskRunner` to run work in the background with progress updates.

```python
from ida_taskr import TaskRunner

def my_heavy_work(data):
    """Your CPU-intensive function."""
    for i, item in enumerate(data):
        # Do work...

        # Send progress update
        yield {
            'type': 'progress',
            'progress': int((i + 1) / len(data) * 100),
            'message': f'Processing {i+1}/{len(data)}'
        }

    return {'result': 'done!'}

# Run it
runner = TaskRunner()

@runner.message_emitter.on('progress')
def on_progress(data):
    print(f"{data['progress']}%: {data['message']}")

@runner.message_emitter.on('result')
def on_result(data):
    print(f"Done: {data['result']}")

runner.run_task(my_heavy_work, data=[1, 2, 3, 4, 5])
# UI is NOT blocked! Progress updates arrive via Qt signals
```

## Examples

### 1. `simple_progress_example.py` - Start Here
The minimal example showing the core pattern:
- Define a worker function with `yield` for progress
- Connect to signals for updates
- Start task with `runner.run_task()`

**When to use:** You need a simple, high-level API for background tasks.

### 2. `signature_generation_example.py` - Complete Guide
Comprehensive examples showing three approaches:

#### Approach 1: TaskRunner (Recommended)
- **Best for:** IDA plugins, most use cases
- **Pros:** Automatic progress updates, easy signal connections
- **Example:** Generating binary signatures with real-time progress

#### Approach 2: ThreadExecutor
- **Best for:** I/O-bound tasks, database queries, network requests
- **Pros:** Lower overhead than processes, shared memory
- **Example:** Running multiple I/O tasks in parallel

#### Approach 3: ProcessPoolExecutor
- **Best for:** CPU-intensive tasks, true parallelism needed
- **Pros:** Bypasses Python GIL, uses multiple CPU cores
- **Example:** Batch processing multiple signatures in parallel

## Key Concepts

### 1. Worker Functions

Worker functions run in separate processes/threads:

```python
def worker(arg1, arg2):
    """Worker function that does heavy lifting."""

    # Send progress updates
    yield {'type': 'progress', 'progress': 50}

    # Return final result
    return {'result': 'value'}
```

### 2. Progress Updates

Use `yield` to send updates without blocking:

```python
for i in range(100):
    # Do work...

    # Send update (becomes Qt signal)
    yield {
        'type': 'progress',
        'progress': i,
        'message': f'Step {i}/100'
    }
```

### 3. Signal Connections

Connect to signals to receive updates:

```python
@runner.message_emitter.on('progress')
def on_progress(data):
    # Update IDA's UI
    idaapi.replace_wait_box(f"{data['message']}")

@runner.message_emitter.on('result')
def on_result(data):
    # Show final result
    print(f"Done: {data['result']}")
```

## Real-World Use Cases

### 1. Binary Signature Generation (8MB file, ~2 minutes)

```python
def generate_unique_signature(start_ea, end_ea):
    """Generate signature for a code range."""
    data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)
    signature = []

    for i in range(min(64, len(data))):
        # Analyze byte for uniqueness
        byte_val = data[i]

        # Check if pattern is unique in database
        if i % 8 == 0:
            yield {
                'type': 'progress',
                'progress': int(i / 64 * 100),
                'message': f'Testing signature at {len(signature)} bytes'
            }

        signature.append(byte_val)

        # Test uniqueness (CPU-intensive)
        if is_unique_pattern(signature):
            break

    return {'signature': signature}

runner = TaskRunner()
runner.run_task(generate_unique_signature, 0x401000, 0xC01000)
# IDA UI stays responsive for 2+ minutes!
```

### 2. Batch Function Analysis

```python
def analyze_all_functions():
    """Analyze every function in the binary."""
    funcs = list(idautils.Functions())

    for i, func_ea in enumerate(funcs):
        # Analyze function (CPU-intensive)
        result = analyze_function(func_ea)

        yield {
            'type': 'progress',
            'progress': int((i + 1) / len(funcs) * 100),
            'message': f'Analyzed {i+1}/{len(funcs)} functions',
            'partial_result': result
        }

    return {'total': len(funcs)}
```

### 3. Parallel Pattern Matching

```python
# Use ProcessPoolExecutor for parallel search
executor = ProcessPoolExecutor(max_workers=4)

patterns = [b'\x55\x8B\xEC', b'\x48\x89\x5C\x24', ...]

futures = []
for pattern in patterns:
    future = executor.submit(search_pattern, pattern, binary_data)
    futures.append(future)

# All searches run in parallel across 4 CPU cores!
```

## Choosing the Right Approach

| Scenario | Use | Why |
|----------|-----|-----|
| Need progress updates | `TaskRunner` | Built-in progress support |
| CPU-intensive, single task | `TaskRunner` | Simple API, automatic process management |
| CPU-intensive, parallel | `ProcessPoolExecutor` | Multiple cores, true parallelism |
| I/O-bound (network, disk) | `ThreadExecutor` | Lower overhead, shared memory |
| Quick one-off task | `create_worker()` | Simplest API |

## Common Patterns

### Pattern 1: Progress Bar in IDA

```python
def long_analysis(data):
    total = len(data)
    for i, item in enumerate(data):
        # Work...
        yield {
            'type': 'progress',
            'progress': int((i + 1) / total * 100)
        }
    return result

runner = TaskRunner()

@runner.message_emitter.on('progress')
def update_progress(data):
    idaapi.replace_wait_box(f"Analyzing: {data['progress']}%")

idaapi.show_wait_box("Starting analysis...")
runner.run_task(long_analysis, data)
```

### Pattern 2: Cancellable Task

```python
runner = TaskRunner()

# User can cancel
@runner.message_emitter.on('cancelled')
def on_cancel(data):
    print("User cancelled the operation")

# Show cancel button in IDA
idaapi.show_wait_box("HIDECANCEL\nAnalyzing...")
runner.run_task(long_task)
```

### Pattern 3: Partial Results

```python
def incremental_search(pattern, data):
    """Return results as they're found."""
    for i, chunk in enumerate(chunks(data)):
        matches = find_matches(pattern, chunk)

        if matches:
            # Send partial results immediately
            yield {
                'type': 'partial_result',
                'matches': matches,
                'chunk': i
            }

    return {'done': True}

@runner.message_emitter.on('partial_result')
def on_match(data):
    # Update UI with each match found
    for match in data['matches']:
        print(f"Found at: 0x{match:X}")
```

## Performance Tips

1. **Use ProcessPoolExecutor for CPU-bound work**
   - Bypasses Python's GIL
   - True parallel execution on multiple cores

2. **Use ThreadExecutor for I/O-bound work**
   - Lower overhead
   - Can share memory with main process

3. **Batch progress updates**
   - Don't yield on every iteration
   - Update every N items or every X seconds

4. **Return early when possible**
   - Don't process more than needed
   - Use `return` as soon as result is found

## Debugging

Enable debug logging:

```python
import logging
from ida_taskr import get_logger

logger = get_logger()
logger.setLevel(logging.DEBUG)
```

## Running the Examples

```bash
# Simple example
python examples/simple_progress_example.py

# Complete examples
python examples/signature_generation_example.py

# In IDA Pro
# File -> Script file... -> Select example file
```

## Next Steps

1. Start with `simple_progress_example.py`
2. Read `signature_generation_example.py` for advanced patterns
3. Check the main README for API documentation
4. Look at tests for more examples
