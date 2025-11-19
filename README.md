# ida-taskr

![CI Status](https://github.com/mahmoudimus/ida-taskr/actions/workflows/python.yml/badge.svg)

## Overview

IDA Taskr is a pure Python library for IDA Pro parallel computing. It lets you use the power of Qt (built-in to IDA!) and Python's multiprocessing to offload computationally intensive tasks to worker processes without freezing IDA Pro's UI.

**Key Features:**
- 🚀 Offload heavy processing to worker processes
- 🔄 Bidirectional IPC communication between IDA and workers
- 📦 Qt-based process management (QProcess)
- 🎯 Event-driven message handling
- ⚡ Compatible with IDA Pro 9.1 (PyQt5) and 9.2+ (PySide6)

## Installation

```bash
# Install from source
pip install -e .

# With Qt support (choose based on your IDA version)
pip install -e .[pyqt5]    # For IDA Pro 9.1
pip install -e .[pyside6]  # For IDA Pro 9.2+
```

## Quick Start

### Basic Example

Here's a simple example of using `TaskRunner` to offload work to a worker process:

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

### Docker Development Environment

For IDA Pro development, use the provided Docker services:

```bash
# Start IDA Pro 9.1 (PyQt5)
docker compose up idapro-91

# Start IDA Pro 9.2 (PySide6)
docker compose up idapro-92
```

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
