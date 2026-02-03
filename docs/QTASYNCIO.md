# QtAsyncio Integration Guide

IDA Taskr now includes **QtAsyncio**, a comprehensive module that integrates Python's `asyncio` with Qt's event loop, providing powerful worker utilities and seamless async/await support in Qt applications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Components](#components)
  - [Worker Utilities](#worker-utilities)
  - [Thread Executor](#thread-executor)
  - [Asyncio Event Loop Integration](#asyncio-event-loop-integration)
- [Usage Examples](#usage-examples)
- [Integration with IDA Taskr](#integration-with-ida-taskr)
- [API Reference](#api-reference)

## Overview

The QtAsyncio module provides three main categories of functionality:

1. **Worker Utilities**: Decorator-based and class-based workers for running tasks in background threads
2. **Thread Executor**: Qt-native `concurrent.futures.Executor` implementation
3. **Asyncio Integration**: Qt-compatible event loop policy for using `async`/`await` in Qt applications

## Features

- ✅ **Thread Workers**: Run functions in background threads with Qt signal integration
- ✅ **Generator Workers**: Support for generator functions that yield intermediate results
- ✅ **Thread Executor**: Qt-based implementation of `concurrent.futures.Executor`
- ✅ **Asyncio Integration**: Seamlessly mix `asyncio` and Qt event loops
- ✅ **Type Hints**: Full type annotation support for better IDE integration
- ✅ **Zero Dependencies**: No external dependencies beyond PyQt5/PySide6
- ✅ **Compatible**: Works with both PyQt5 (IDA 9.1) and PySide6 (IDA 9.2+)

## Quick Start

### Check Availability

```python
from ida_taskr import QT_ASYNCIO_AVAILABLE

if QT_ASYNCIO_AVAILABLE:
    print("QtAsyncio is available!")
```

### Simple Worker Example

```python
from ida_taskr import thread_worker

@thread_worker
def compute_something(x, y):
    # This runs in a background thread
    return x + y

# In your Qt application:
worker = compute_something(10, 20)
worker.returned.connect(lambda result: print(f"Result: {result}"))
worker.start()
```

### Asyncio Integration Example

```python
import asyncio
from ida_taskr import set_event_loop_policy

# Enable Qt-asyncio integration
set_event_loop_policy()

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

# Now you can use async/await in your Qt application!
asyncio.create_task(fetch_data())
```

## Components

### Worker Utilities

The worker utilities provide an easy way to run tasks in background threads without blocking the Qt UI.

#### FunctionWorker

For regular functions:

```python
from ida_taskr import create_worker

def long_task(duration):
    import time
    time.sleep(duration)
    return f"Completed after {duration}s"

worker = create_worker(long_task, 5)
worker.returned.connect(on_result)
worker.errored.connect(on_error)
worker.finished.connect(on_finished)
worker.start()
```

#### GeneratorWorker

For generator functions that yield intermediate results:

```python
from ida_taskr import create_worker

def process_items(items):
    for item in items:
        # Do some work
        result = process(item)
        yield result

worker = create_worker(process_items, my_items)
worker.yielded.connect(on_intermediate_result)
worker.returned.connect(on_complete)
worker.start()
```

#### The `@thread_worker` Decorator

The simplest way to create workers:

```python
from ida_taskr import thread_worker

@thread_worker
def my_background_task(x):
    # Heavy computation here
    return x * 2

# Creates and returns a worker
worker = my_background_task(100)
worker.start()
```

#### Worker Signals

All workers provide these signals:

- `started` - Emitted when the worker starts
- `finished` - Emitted when the worker finishes (success or error)
- `returned(result)` - Emitted with the return value
- `errored(exception)` - Emitted if an exception occurs
- `warned(warning)` - Emitted for warnings

Generator workers additionally provide:

- `yielded(value)` - Emitted for each yielded value
- `paused` - Emitted when paused
- `resumed` - Emitted when resumed
- `aborted` - Emitted when aborted

### Thread Executor

Qt-native implementation of `concurrent.futures.Executor`:

```python
from ida_taskr import ThreadExecutor

executor = ThreadExecutor()

def cpu_intensive(n):
    return sum(i*i for i in range(n))

future = executor.submit(cpu_intensive, 1000000)
future.add_done_callback(lambda f: print(f.result()))

# Clean up when done
executor.shutdown(wait=True)
```

### Asyncio Event Loop Integration

Enable seamless integration between asyncio and Qt:

```python
import asyncio
from ida_taskr import set_event_loop_policy

# Call this once at application startup
set_event_loop_policy()

# Now you can use asyncio naturally with Qt
async def my_async_function():
    result = await asyncio.sleep(1)
    return "done"

# In your Qt application
asyncio.create_task(my_async_function())
```

## Usage Examples

### Example 1: Progress Updates

```python
from ida_taskr import create_worker

def process_with_progress(items):
    total = len(items)
    for i, item in enumerate(items):
        # Process item
        result = do_work(item)
        # Yield progress
        yield {"progress": (i + 1) / total, "item": item}
    return "All done!"

worker = create_worker(process_with_progress, my_items)

def on_progress(data):
    print(f"Progress: {data['progress']*100:.1f}%")

worker.yielded.connect(on_progress)
worker.returned.connect(lambda r: print(f"Complete: {r}"))
worker.start()
```

### Example 2: Error Handling

```python
from ida_taskr import thread_worker

@thread_worker
def risky_operation(data):
    if not data:
        raise ValueError("Data cannot be empty")
    return process_data(data)

worker = risky_operation(my_data)

def on_error(exception):
    print(f"Error occurred: {exception}")
    # Handle error appropriately

worker.errored.connect(on_error)
worker.returned.connect(on_success)
worker.start()
```

### Example 3: Mixing Asyncio and Qt

```python
import asyncio
from ida_taskr import set_event_loop_policy
from PySide6.QtWidgets import QPushButton

set_event_loop_policy()

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.button = QPushButton("Fetch Data")
        self.button.clicked.connect(self.on_click)

    def on_click(self):
        # Start an async task from a Qt signal
        asyncio.create_task(self.fetch_and_update())

    async def fetch_and_update(self):
        data = await self.fetch_data_async()
        self.update_ui(data)

    async def fetch_data_async(self):
        # Simulate async I/O
        await asyncio.sleep(2)
        return {"status": "success"}
```

## Integration with IDA Taskr

The QtAsyncio module integrates seamlessly with IDA Taskr's existing components:

### Using QtAsyncio with WorkerController

```python
from ida_taskr.worker import WorkerController
from ida_taskr.utils import AsyncEventEmitter

class MyEmitter(AsyncEventEmitter):
    async def run(self):
        # Your async work here
        return result

emitter = MyEmitter()

# Enable QtAsyncio integration
controller = WorkerController(emitter, use_qtasyncio=True)
controller.start()
```

### When to Use QtAsyncio

**Use QtAsyncio Worker Utilities when:**
- You want simple background task execution with Qt signals
- You need to update the UI from worker threads
- You want a decorator-based API

**Use IDA Taskr's WorkerBase when:**
- You need multiprocessing (separate process isolation)
- You're building IDA Pro plugins with heavy computation
- You need bidirectional IPC between IDA and worker processes

**Use QtAsyncio Event Loop when:**
- You want to use `async`/`await` in your Qt application
- You're mixing asyncio libraries with Qt
- You need modern Python async patterns

## API Reference

### Worker Utilities

#### `create_worker(func, *args, **kwargs)`

Create a worker from a function or generator function.

**Returns:** `FunctionWorker` or `GeneratorWorker` depending on function type

#### `@thread_worker(func)`

Decorator to create a worker-returning function.

**Example:**
```python
@thread_worker
def my_func(x):
    return x * 2

worker = my_func(10)  # Returns a worker
worker.start()
```

#### `new_worker_qthread(Worker, *args, _start_thread=False, _connect=None, **kwargs)`

Create a QThread-based worker (alternative to QRunnable).

**Parameters:**
- `Worker`: QObject subclass with a `work()` method
- `_start_thread`: Whether to start the thread immediately
- `_connect`: Dict of signal->slot connections

**Returns:** `(worker, thread)` tuple

### Thread Executor

#### `ThreadExecutor(parent=None, threadPool=None)`

Qt-native concurrent.futures.Executor.

**Methods:**
- `submit(func, *args, **kwargs)`: Submit a callable for execution
- `shutdown(wait=True)`: Shutdown the executor

### Asyncio Integration

#### `set_event_loop_policy()`

Set the Qt-compatible asyncio event loop policy.

#### `QAsyncioEventLoop(asyncio_loop, parent=None)`

Qt event loop that integrates with asyncio.

#### `QAsyncioEventLoopPolicy`

Event loop policy that creates Qt-integrated event loops.

## Best Practices

1. **Always check availability:**
   ```python
   from ida_taskr import QT_ASYNCIO_AVAILABLE
   if not QT_ASYNCIO_AVAILABLE:
       # Fallback code
   ```

2. **Set event loop policy early:**
   ```python
   # At application startup
   set_event_loop_policy()
   ```

3. **Clean up workers:**
   ```python
   worker.finished.connect(worker.deleteLater)
   ```

4. **Handle errors:**
   ```python
   worker.errored.connect(handle_error)
   ```

5. **Use appropriate worker type:**
   - CPU-bound tasks → `FunctionWorker`
   - Tasks with progress → `GeneratorWorker`
   - Heavy IDA analysis → IDA Taskr's `WorkerBase` with multiprocessing

## Examples

Complete examples are available in the `examples/` directory:

- `thread_worker_usage.py` - Demonstrates worker utilities
- `qtasyncio_event_loop.py` - Shows asyncio event loop integration

## Troubleshooting

**Q: Import error when importing qtasyncio**
- Ensure PyQt5 or PySide6 is installed
- Check `QT_ASYNCIO_AVAILABLE` before importing

**Q: Workers don't emit signals**
- Make sure you call `worker.start()`
- Ensure Qt event loop is running

**Q: Async tasks don't run**
- Call `set_event_loop_policy()` before creating tasks
- Ensure Qt application is running

## License

This module is based on Qt's official asyncio integration (commit: 072ffd057a29a694a0ad91894736bb4d0a88738e) with additional utilities and enhancements for IDA Taskr.
