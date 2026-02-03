# ida-taskr

![CI Status](https://github.com/mahmoudimus/ida-taskr/actions/workflows/python.yml/badge.svg)

## Overview

IDA Taskr is a pure Python library for IDA Pro parallel computing. It lets you use the power of Qt (built-in to IDA!) and Python's multiprocessing to offload computationally intensive tasks to worker processes without freezing IDA Pro's UI.

**Key Features:**
- 🚀 Simple decorator API - just add `@cpu_task` to run in background
- 🔄 Process-based parallelism for true multi-core execution
- 📦 Shared memory support for large binary data
- 🎯 Qt signal integration for progress callbacks
- ⚡ Compatible with IDA Pro 9.1 (PyQt5) and 9.2+ (PySide6)

## Installation

**Option 1: Single file (no install needed)**

Download [`ida_taskr_amalgamated.py`](https://github.com/mahmoudimus/ida-taskr/releases/latest) and drop into IDA's plugins folder. That's it - one file, zero dependencies!

**Option 2: pip install**
```bash
pip install ida-taskr

# Or from source with Qt support
pip install -e .[pyqt5]    # For IDA Pro 9.1
pip install -e .[pyside6]  # For IDA Pro 9.2+
```

**Option 3: IDA Plugin Manager (HCLI)**

Download [`ida-taskr-{version}.zip`](https://github.com/mahmoudimus/ida-taskr/releases/latest) and install via HCLI.

## Quick Start

### The Simplest Way: `@cpu_task`

Add one decorator and your function runs in the background:

```python
from ida_taskr import cpu_task

@cpu_task
def analyze_binary(data):
    """This runs in a background thread - UI stays responsive!"""
    result = []
    for byte in data:
        result.append(process_byte(byte))
    return result

# Usage - returns immediately!
future = analyze_binary(binary_data)

# Do other work while it runs...

# Get result when needed
result = future.result()
```

That's it. One line. Your function now runs without blocking IDA.

### With Callbacks

Get notified when your task completes:

```python
from ida_taskr import cpu_task

@cpu_task(on_complete=lambda r: print(f"Done! Found {len(r)} patterns"))
def find_patterns(data):
    return scan_for_patterns(data)

# Fire and forget - callback handles the result
find_patterns(binary_data)
```

### Parallel Processing

Process multiple items across worker threads:

```python
from ida_taskr import parallel

@parallel(max_workers=8)
def analyze_function(func_ea):
    """Analyze a single function."""
    return get_function_signature(func_ea)

# Process 100 functions in parallel
function_addresses = list(idautils.Functions())
futures = [analyze_function(addr) for addr in function_addresses]
results = [f.result() for f in futures]
```

### Large Data with Shared Memory

For large binary blobs (megabytes), use shared memory to avoid copying:

```python
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def find_signatures(chunk_data, chunk_id, total_chunks):
    """
    Process one chunk of the binary.

    chunk_data: memoryview of this chunk (zero-copy!)
    chunk_id: which chunk this is (0-7)
    total_chunks: total number of chunks (8)
    """
    signatures = []
    for i in range(len(chunk_data) - 16):
        if is_interesting_pattern(chunk_data[i:i+16]):
            signatures.append(bytes(chunk_data[i:i+16]))
    return signatures

# ida-taskr handles all shared memory complexity!
binary_data = ida_bytes.get_bytes(start_ea, size)  # e.g., 8MB
all_signatures = find_signatures(binary_data)  # Returns list of 8 results
```

## Decorator Reference

| Decorator | Use Case | Example |
|-----------|----------|---------|
| `@cpu_task` | CPU-intensive work | Pattern scanning, signature generation |
| `@io_task` | I/O-bound work | Network requests, file operations |
| `@parallel(n)` | Multiple parallel tasks | Batch function analysis |
| `@background_task` | Full control with callbacks | Progress reporting |
| `@shared_memory_task` | Large data processing | Multi-MB binary analysis |

### `@background_task` - Full Control

```python
from ida_taskr import background_task

@background_task(
    max_workers=4,
    on_complete=lambda r: print(f"Result: {r}"),
    on_error=lambda e: print(f"Error: {e}"),
    on_progress=lambda p, m: print(f"[{p}%] {m}"),
    executor_type='process'  # 'thread' or 'process'
)
def heavy_analysis(data, progress_callback=None):
    for i, chunk in enumerate(chunks(data, 100)):
        process(chunk)
        if progress_callback:
            progress_callback(i * 10, f"Processed chunk {i}")
    return "done"
```

## Advanced Usage

### Direct Executor Access

For more control, use the executors directly:

```python
from ida_taskr import ProcessPoolExecutor, ThreadExecutor

# Process-based (true parallelism, bypasses GIL)
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(cpu_task, arg) for arg in args]
    results = [f.result() for f in futures]

# Thread-based (good for IDA SDK calls that release GIL)
with ThreadExecutor(max_workers=8) as executor:
    futures = [executor.submit(analyze_func, ea) for ea in function_list]
    results = [f.result() for f in futures]
```

### Worker Scripts (Bidirectional IPC)

For complex scenarios requiring persistent workers and bidirectional communication.
Use this when you need:
- Long-running worker processes that stay alive between tasks
- Custom message protocols between IDA and workers
- Streaming results back to IDA as work progresses
- Worker state that persists across multiple commands

```python
from ida_taskr import TaskRunner

runner = TaskRunner(
    worker_script="path/to/worker.py",
    worker_args=["arg1", "arg2"]
)

@runner.on('worker_results')
def handle_results(results):
    print(f"Results: {results}")

@runner.on('worker_message')
def handle_progress(msg):
    print(f"Progress: {msg}")

runner.start()
runner.send_command({"command": "process", "data": [1, 2, 3]})
# Worker stays alive, can send more commands...
runner.send_command({"command": "analyze", "target": 0x401000})
runner.stop()
```

See [examples/](examples/) for more detailed examples including:
- [Ultra minimal example](examples/ultra_minimal.py) - Smallest possible code
- [Shared memory patterns](examples/shared_memory_parallel_example.py) - Large data processing
- [Signature generation](examples/signature_generation_example.py) - Real IDA use case
- [QtAsyncio integration](examples/qtasyncio_event_loop.py) - Async/await support

## Testing

```bash
# Run all unit tests
./run_tests.sh

# Run Qt integration tests
pytest tests/integration/test_integration_qt_core.py -v

# Run with coverage
pytest tests/ --cov=src/ida_taskr --cov-report=html
```

**Supported Configurations:**
- ✅ Python 3.11, 3.12, 3.13
- ✅ PyQt5 (IDA Pro 9.1)
- ✅ PySide6 (IDA Pro 9.2+)

## Documentation

- [QtAsyncio Integration](docs/QTASYNCIO.md) - Async/await and event loop details
- [IDA Testing Guide](docs/IDA_TESTING.md) - Running tests inside IDA Pro
- [Examples README](examples/README.md) - Comprehensive examples guide

## Contributing 🤝

We welcome contributions! See the examples and tests for code style.

1. Fork the repository
2. Create a feature branch
3. Run tests: `./run_tests.sh`
4. Submit a pull request

## License 📜

MIT License - see [LICENSE](LICENSE) for details.

## Contact 📧

Questions or issues? Open a GitHub issue or reach out to [@mahmoudimus](https://github.com/mahmoudimus).
