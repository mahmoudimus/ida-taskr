# ida-taskr Examples

Complete examples demonstrating how to use ida-taskr for background task processing in IDA Pro.

---

## 🚀 Quick Start: The Simplest Way

**Problem:** You need to do CPU-intensive work without freezing IDA's UI.

**Solution:** Just add `@cpu_task` to your function!

```python
from ida_taskr import cpu_task

@cpu_task
def analyze(data):
    # Your CPU-intensive code
    return result

# Returns immediately - runs in background!
future = analyze(data)
result = future.result()
```

**That's it! One line decorator and done.**

---

## 📚 What's in This Directory?

### ⭐ **Start Here: Minimal API Examples**

These show the **absolute simplest** way to use ida-taskr:

| File | What It Shows | Lines of Code |
|------|---------------|---------------|
| **[one_line_solution.py](one_line_solution.py)** | 🏆 The one-line answer: `@cpu_task` | **1 line!** |
| **[ultra_minimal.py](ultra_minimal.py)** | Absolute minimal working example | ~10 lines |
| **[api_simplicity_levels.py](api_simplicity_levels.py)** | All API levels (verbose → minimal) | Comparison |
| **[README_MINIMAL_API.md](README_MINIMAL_API.md)** | Complete decorator documentation | Guide |

### 🚀 **Shared Memory Examples (For 8MB+ Data)**

These show how to process large binary data efficiently:

| File | What It Shows | Reduction |
|------|---------------|-----------|
| **[shared_memory_one_line.py](shared_memory_one_line.py)** | 🏆 Minimal shared memory API | **8x simpler!** |
| **[shared_memory_decorator_example.py](shared_memory_decorator_example.py)** | Working 8MB example | 40 → 5 lines |
| **[shared_memory_comparison.py](shared_memory_comparison.py)** | Before/after comparison | Side-by-side |
| **[ida_shared_memory_pattern.py](ida_shared_memory_pattern.py)** | IDA-specific pattern (anti_deob style) | Production |
| **[README_SHARED_MEMORY.md](README_SHARED_MEMORY.md)** | Complete shared memory docs | Guide |

### 💻 **Real-World CPU-Intensive Examples**

| File | What It Shows |
|------|---------------|
| **[decorator_simple_example.py](decorator_simple_example.py)** | Binary signature generation |
| **[decorator_evolution.py](decorator_evolution.py)** | Blocking → Async evolution |
| **[cpu_intensive_example.py](cpu_intensive_example.py)** | Complete ProcessPoolExecutor example |
| **[cpu_intensive_with_progress.py](cpu_intensive_with_progress.py)** | With progress updates |

### 📊 **Classic TaskRunner Examples**

| File | What It Shows |
|------|---------------|
| **simple_progress_example.py** | Basic TaskRunner with progress |
| **signature_generation_example.py** | Complete guide with 3 approaches |

---

## 🎯 Example Selection Guide

### "I just want the simplest way to make my function non-blocking"
→ **[one_line_solution.py](one_line_solution.py)** - Just add `@cpu_task`!

### "I need to process large binary data (8MB+) efficiently"
→ **[shared_memory_one_line.py](shared_memory_one_line.py)** - Zero-copy shared memory

### "I want to understand all the API options"
→ **[api_simplicity_levels.py](api_simplicity_levels.py)** - See all 5 levels

### "I need a real-world IDA example"
→ **[decorator_simple_example.py](decorator_simple_example.py)** - Binary signatures
→ **[ida_shared_memory_pattern.py](ida_shared_memory_pattern.py)** - Large data pattern

### "I want comprehensive documentation"
→ **[README_MINIMAL_API.md](README_MINIMAL_API.md)** - Decorator guide
→ **[README_SHARED_MEMORY.md](README_SHARED_MEMORY.md)** - Shared memory guide

---

## 🔥 Key Decorators Reference

### `@cpu_task` - Simplest for CPU work

```python
from ida_taskr import cpu_task

@cpu_task
def analyze(data):
    # CPU-intensive work
    return result

future = analyze(data)  # Returns immediately
result = future.result()  # Get result when ready
```

**Use when:** You want the simplest way to run CPU work without blocking.

### `@cpu_task` with callback

```python
@cpu_task(on_complete=lambda r: print(f"Done: {r}"))
def analyze(data):
    return result

analyze(data)  # Fire and forget - callback handles result
```

**Use when:** You want automatic result delivery.

### `@shared_memory_task` - For large data (8MB+)

```python
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze_chunk(chunk_data, chunk_id, total_chunks):
    # Process this chunk (no data copying!)
    return find_patterns(chunk_data)

# Processes 8MB in 8 parallel chunks
results = analyze_chunk(binary_data)
```

**Use when:** Processing large binary data where copying would be expensive.

### `@background_task` - Full control

```python
from ida_taskr import background_task

@background_task(
    max_workers=4,
    on_complete=handle_result,
    on_error=handle_error,
    executor_type='thread'  # or 'process'
)
def task(args):
    return result
```

**Use when:** You need fine-grained control over all options.

---

## 📖 Classic TaskRunner API

For progress-based workflows with yield:

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

---

## 💡 Why Use ida-taskr?

### ❌ Without ida-taskr (Manual Setup)

```python
from concurrent.futures import ProcessPoolExecutor

def analyze(data):
    def worker(data):
        # Your code
        return result

    executor = ProcessPoolExecutor(max_workers=4)
    try:
        future = executor.submit(worker, data)
        result = future.result(timeout=10)
        return result
    finally:
        executor.shutdown(wait=True)

# ~15 lines of boilerplate for EACH function!
```

### ✅ With ida-taskr (Decorator)

```python
from ida_taskr import cpu_task

@cpu_task
def analyze(data):
    # Your code
    return result

# Just 1 line!
```

---

## ⚡ Performance Benefits

### Shared Memory Pattern

**Traditional multiprocessing:**
- 8MB data × 8 processes = **64MB copied**
- High memory usage
- Slow serialization

**With `@shared_memory_task`:**
- 8MB data copied **ONCE**
- Workers attach via name (zero-copy)
- **~8x faster** for large data!

### Real-World Benchmark

```python
# Process 8MB binary in IDA
binary_data = ida_bytes.get_bytes(start_ea, 8 * 1024 * 1024)

# Traditional approach: ~500ms (copying + processing)
# With @shared_memory_task: ~60ms (no copying!)

@shared_memory_task(num_chunks=8)
def find_patterns(chunk_data, chunk_id, total_chunks):
    return analyze(chunk_data)

results = find_patterns(binary_data)  # 8x faster!
```

---

## 🎨 Common Patterns

### Pattern 1: Simple Background Task

```python
@cpu_task
def find_functions(data):
    return analyze(data)

future = find_functions(binary_data)
result = future.result()
```

**Lines: 1 decorator + your function**

### Pattern 2: Background Task with Callback

```python
@cpu_task(on_complete=show_results)
def find_functions(data):
    return analyze(data)

find_functions(binary_data)  # Auto-delivered to show_results()
```

**Lines: 1 decorator + callback**

### Pattern 3: Shared Memory for Large Data

```python
@shared_memory_task(num_chunks=8)
def analyze_chunk(chunk_data, chunk_id, total_chunks):
    return find_patterns(chunk_data)

results = analyze_chunk(binary_data)  # 8 chunks in parallel
```

**Lines: 1 decorator + chunk logic**

### Pattern 4: Batch Processing

```python
@parallel(max_workers=8)
def process_function(address):
    return analyze_function(address)

futures = [process_function(addr) for addr in function_list]
results = [f.result() for f in futures]
```

**Lines: 1 decorator + your function**

---

## 📊 Complexity Reduction

| Pattern | Manual | With Decorator | Reduction |
|---------|--------|----------------|-----------|
| **Simple task** | ~15 lines | 1 line | **15x** |
| **Shared memory** | ~40-50 lines | ~5 lines | **8-10x** |
| **With callbacks** | ~20 lines | 2 lines | **10x** |

---

## 🚀 Running the Examples

### Prerequisites

```bash
# Install ida-taskr
pip install -e .

# Install Qt (PyQt5 for IDA ≤9.1, PySide6 for IDA ≥9.2)
pip install PyQt5  # or PySide6
```

### Running Examples Standalone

```bash
# Minimal API examples
python examples/one_line_solution.py
python examples/ultra_minimal.py
python examples/api_simplicity_levels.py

# Shared memory examples
python examples/shared_memory_one_line.py
python examples/shared_memory_decorator_example.py

# See all API levels
python examples/api_simplicity_levels.py
```

### Running in IDA Pro

In IDA's Python console:

```python
# Load example
import sys
sys.path.insert(0, '/path/to/ida-taskr/examples')

# Use decorator
from ida_taskr import cpu_task

@cpu_task
def analyze_current_function():
    import ida_bytes
    ea = idc.here()
    data = ida_bytes.get_bytes(ea, 1024)
    # Your analysis
    return result

future = analyze_current_function()
result = future.result()
```

Or use `File → Script file...` to run example files directly.

---

## 🔧 Debugging

Enable debug logging:

```python
import logging
from ida_taskr import get_logger

logger = get_logger()
logger.setLevel(logging.DEBUG)
```

---

## 📝 Summary

### Start With These Files

1. **[one_line_solution.py](one_line_solution.py)** - Simplest background task (1 line!)
2. **[shared_memory_one_line.py](shared_memory_one_line.py)** - Simplest shared memory (~5 lines)
3. **[decorator_simple_example.py](decorator_simple_example.py)** - Real IDA example

### Then Read the Guides

- **[README_MINIMAL_API.md](README_MINIMAL_API.md)** - Complete decorator documentation
- **[README_SHARED_MEMORY.md](README_SHARED_MEMORY.md)** - Complete shared memory guide

### Complexity Levels

| What You Write | Manual Setup | With Decorator |
|----------------|--------------|----------------|
| **Simple task** | ~15 lines | **1 line** |
| **Shared memory** | ~40 lines | **~5 lines** |
| **With callbacks** | ~20 lines | **2 lines** |

---

## 🎯 Choosing the Right Approach

| Scenario | Use | Example File |
|----------|-----|--------------|
| **Simple background task** | `@cpu_task` | [one_line_solution.py](one_line_solution.py) |
| **Large binary data (8MB+)** | `@shared_memory_task` | [shared_memory_one_line.py](shared_memory_one_line.py) |
| **Need progress updates** | `TaskRunner` | simple_progress_example.py |
| **Parallel batch processing** | `@parallel` | [decorator_simple_example.py](decorator_simple_example.py) |
| **Full control** | `@background_task` | [api_simplicity_levels.py](api_simplicity_levels.py) |

---

## 📚 Next Steps

1. **Quick start:** Run [one_line_solution.py](one_line_solution.py) to see the minimal API
2. **Learn levels:** Run [api_simplicity_levels.py](api_simplicity_levels.py) to see all options
3. **Large data:** Read [README_SHARED_MEMORY.md](README_SHARED_MEMORY.md) for shared memory
4. **Deep dive:** Read [README_MINIMAL_API.md](README_MINIMAL_API.md) for complete reference
5. **Real examples:** Check [decorator_simple_example.py](decorator_simple_example.py) and [ida_shared_memory_pattern.py](ida_shared_memory_pattern.py)

---

**Happy coding! 🎉**
