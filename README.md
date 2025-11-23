# ida-taskr

![CI Status](https://github.com/mahmoudimus/ida-taskr/actions/workflows/python.yml/badge.svg)

## Overview

IDA Taskr is a pure Python library for IDA Pro parallel computing. It lets you use the power of Qt (built-in to IDA!) and Python's multiprocessing to offload computationally intensive tasks to worker processes without freezing IDA Pro's UI.

**Key Features:**
- 🚀 **One-line decorators** for instant parallelism (`@cpu_task`)
- 🔥 **SharedMemoryExecutor** with zero-copy workers (8x faster for large data)
- 🎯 **Pure function signatures** - no chunk-specific parameters needed
- 📦 **Standard `concurrent.futures` interface** - familiar and powerful
- 🔄 **Qt signal integration** for progress tracking
- ⚡ **Compatible with IDA Pro 9.1** (PyQt5) and 9.2+ (PySide6)

## Installation

```bash
# Install from source
pip install -e .

# With Qt support (choose based on your IDA version)
pip install -e .[pyqt5]    # For IDA Pro 9.1
pip install -e .[pyside6]  # For IDA Pro 9.2+
```

## Quick Start

### ⚡ Simplest Way: One-Line Decorator

Process large data in parallel with just one decorator:

```python
from ida_taskr import cpu_task

@cpu_task
def analyze_binary(data):
    # Your CPU-intensive code here
    return find_patterns(data)

# Returns immediately - runs in background!
future = analyze_binary(binary_data)
result = future.result()
```

### 🚀 Shared Memory for Large Data (8MB+)

For large binary data, use `SharedMemoryExecutor` with **zero-copy** workers:

```python
from ida_taskr import SharedMemoryExecutor

def find_patterns(data):
    # Pure function - doesn't know about chunks!
    return [i for i in range(len(data)) if data[i] == 0xFF]

# Process 8MB in 8 parallel chunks with zero data copying
executor = SharedMemoryExecutor(max_workers=8)
future = executor.submit_chunked(find_patterns, binary_data, num_chunks=8)
results = future.result()  # 8x faster than copying!
```

**Key Benefits:**
- ✅ **Pure function signatures** - `fn(data)` never changes
- ✅ **Standard `concurrent.futures` interface** - familiar API
- ✅ **Zero-copy workers** - attach to shared memory (8x faster)
- ✅ **Qt signal integration** - track progress with signals
- ✅ **Automatic cleanup** - SharedMemory lifecycle managed

### 📚 Classic TaskRunner Example

For progress-based workflows, use `TaskRunner`:

```python
from ida_taskr import TaskRunner

# Create a task runner with your worker script
runner = TaskRunner(
    worker_script="path/to/worker.py",
    worker_args=["arg1", "arg2"]
)

# Set up message handlers
@runner.on('worker_message')
def handle_message(msg):
    print(f"Worker said: {msg}")

@runner.on('worker_results')
def handle_results(results):
    print(f"Results: {results}")

# Start the worker
runner.start()

# Send commands to the worker
runner.send_command({"command": "process", "data": [1, 2, 3]})

# When done
runner.stop()
```

### Worker Script Example

Your worker script receives commands and sends results back:

```python
# worker.py
import sys
from ida_taskr.worker import WorkerBase

class MyWorker(WorkerBase):
    def handle_command(self, command):
        """Process commands from IDA."""
        if command.get("command") == "process":
            data = command.get("data", [])

            # Do heavy computation here
            result = [x * 2 for x in data]

            # Send results back
            self.send_message({
                "type": "result",
                "data": result
            })

if __name__ == "__main__":
    worker = MyWorker()
    worker.run()
```

### Advanced: Using WorkerLauncher Directly

For more control, use `WorkerLauncher`:

```python
from ida_taskr.launcher import WorkerLauncher
from ida_taskr.protocols import MessageEmitter

# Create message emitter for event handling
emitter = MessageEmitter()

@emitter.on('worker_message')
def on_message(msg):
    print(f"Message: {msg}")

@emitter.on('worker_connected')
def on_connected():
    print("Worker connected!")

# Create launcher
launcher = WorkerLauncher(message_emitter=emitter)

# Launch worker process
launcher.launch_worker(
    script_path="worker.py",
    worker_args={"chunk_size": "1024", "mode": "fast"}
)

# Send commands
launcher.send_command({"command": "analyze", "ea": 0x401000})

# Stop when done
launcher.stop_worker()
```

## Testing

`ida-taskr` is thoroughly tested across multiple Qt frameworks and Python versions.

### Unit Tests

Unit tests don't require IDA Pro and use mocks where needed:

```bash
# Run all unit tests
./run_tests.sh

# Or use unittest directly
python -m unittest discover -s tests/unit -p "test_*.py"

# Run a specific test
./run_tests.sh test_event_emitter
```

### Integration Tests

Integration tests verify compatibility with real Qt frameworks:

```bash
# Install test dependencies
pip install -e .[ci,pyqt5]    # For PyQt5 tests
pip install -e .[ci,pyside6]  # For PySide6 tests

# Run Qt integration tests
pytest tests/integration/test_integration_qt_core.py -v

# Run with coverage
pytest tests/integration/ --cov=src/ida_taskr --cov-report=html
```

**Supported Qt Frameworks:**
- ✅ PyQt5 (IDA Pro 9.1)
- ✅ PySide6 (IDA Pro 9.2+)

## API Comparison

Choose the right approach for your use case:

| Approach | Best For | Lines of Code | Function Signature |
|----------|----------|---------------|-------------------|
| **`@cpu_task`** | Quick scripts, simple tasks | 1 line | `fn(data)` |
| **`@shared_memory_task`** | Large data (8MB+), minimal code | ~5 lines | `fn(chunk_data, chunk_id, total)` |
| **`SharedMemoryExecutor`** | Libraries, full control, reusability | ~5-10 lines | `fn(data)` ← **Pure!** |
| **`TaskRunner`** | Progress updates, complex workflows | ~15-20 lines | Custom worker class |

### When to Use SharedMemoryExecutor

✅ **Use SharedMemoryExecutor when:**
- You want **pure function signatures** (reusable across contexts)
- You need **full control** over executor lifecycle
- You want to **reuse executor** for multiple operations
- You need **Qt signal integration** for progress tracking
- You're building **libraries or complex applications**
- You want **standard `concurrent.futures` interface**

✅ **Use `@shared_memory_task` decorator when:**
- You want the **simplest possible API** (just add decorator)
- You're writing **quick scripts or plugins**
- You don't need executor control or signal integration

### SharedMemoryExecutor Features

The `SharedMemoryExecutor` provides a powerful, standard interface for shared memory processing:

```python
from ida_taskr import SharedMemoryExecutor

executor = SharedMemoryExecutor(max_workers=8)

# Standard interface (like ProcessPoolExecutor)
future = executor.submit(func, arg1, arg2)
results = executor.map(func, [item1, item2, ...])

# Shared memory optimization
future = executor.submit_chunked(func, binary_data, num_chunks=8)

# With custom result combining
future = executor.submit_chunked(
    func, data, num_chunks=8,
    combine=lambda results: sum(results, [])  # Flatten lists
)

# Streaming results as chunks complete
for result in executor.map_chunked(func, data, num_chunks=8):
    print(f"Chunk done: {result}")

# Qt signal integration
executor.signals.chunk_completed.connect(on_chunk_done)
executor.signals.all_chunks_completed.connect(on_all_done)
```

**Examples:** See `examples/shared_memory_executor_*.py` for complete examples including IDA Pro integration.

## Contributing 🤝

We welcome contributions to `ida-taskr`! Whether it's bug fixes, new features, or documentation improvements, your help is appreciated. Here's how to contribute:

1. **Fork the Repository** and clone it locally. 🍴
2. **Make Your Changes** in a new branch. 🌿
3. **Run Tests** to ensure everything works (`python3 -m unittest discover -s tests/unit/`). 🧪
4. **Submit a Pull Request** with a clear description of your changes. 📬

Please follow the coding style and include tests for new functionality. Let's make `ida-taskr` even better together! 💪

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 📄

## Contact 📧

Have questions, suggestions, or need support? Open an issue on GitHub or reach out to [mahmoudimus](https://github.com/mahmoudimus). I'm happy to help! 😊
