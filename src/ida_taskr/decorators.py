"""
Decorators for simplifying async task execution.

Provides simple decorator-based API for running CPU-intensive tasks
in the background without blocking.
"""

import functools
from typing import Callable, Optional, Any
from concurrent.futures import Future

from .qt_compat import QT_AVAILABLE

if QT_AVAILABLE:
    from .qtasyncio import ProcessPoolExecutor, ThreadExecutor

# Global executor instance (created on first use)
_global_executor = None


def get_global_executor(max_workers: Optional[int] = None):
    """Get or create the global background task executor."""
    global _global_executor

    if _global_executor is None:
        if QT_AVAILABLE:
            _global_executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise RuntimeError("Qt is not available - cannot create executor")

    return _global_executor


def background_task(
    func: Optional[Callable] = None,
    *,
    max_workers: Optional[int] = None,
    on_complete: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    timeout: Optional[float] = None,
    executor_type: str = 'thread'  # 'thread' (default) or 'process'
):
    """
    Decorator to run a function as a background task without blocking.

    Transforms a regular function into one that runs in a separate process/thread
    and returns a Future immediately.

    Args:
        func: The function to decorate (auto-filled when used without parens)
        max_workers: Maximum number of parallel workers (default: CPU count)
        on_complete: Callback when task completes, receives result
        on_error: Callback when task fails, receives exception
        on_progress: Callback for progress updates (progress%, message)
        timeout: Optional timeout in seconds
        executor_type: 'process' for CPU-bound, 'thread' for I/O-bound

    Returns:
        Decorated function that returns a Future when called

    Examples:
        Basic usage:
        >>> @background_task
        ... def slow_function(x):
        ...     return x * 2
        ...
        >>> future = slow_function(5)  # Returns immediately
        >>> result = future.result()   # Get result when needed

        With callback:
        >>> @background_task(on_complete=lambda r: print(f"Done: {r}"))
        ... def slow_function(x):
        ...     return x * 2
        ...
        >>> slow_function(5)  # Callback fires when done

        With progress:
        >>> @background_task(
        ...     on_progress=lambda p, m: print(f"[{p}%] {m}"),
        ...     on_complete=lambda r: print(f"Result: {r}")
        ... )
        ... def slow_function(x, progress_callback=None):
        ...     if progress_callback:
        ...         progress_callback(50, "Halfway done")
        ...     return x * 2

        Multiple parallel tasks:
        >>> @background_task(max_workers=4)
        ... def process_item(item):
        ...     return expensive_computation(item)
        ...
        >>> futures = [process_item(x) for x in range(100)]
        >>> results = [f.result() for f in futures]
    """

    def decorator(fn: Callable) -> Callable:
        """The actual decorator function."""

        # Create or get executor
        if executor_type == 'thread' and QT_AVAILABLE:
            from .qtasyncio import ThreadExecutor
            executor = ThreadExecutor(max_workers=max_workers)
        else:
            executor = get_global_executor(max_workers=max_workers)

        # Connect callbacks to signals if provided
        if on_complete:
            def handle_complete(future: Future):
                try:
                    result = future.result(timeout=timeout)
                    on_complete(result)
                except Exception:
                    pass  # Error handled by on_error if registered

            executor.signals.task_completed.connect(handle_complete)

        if on_error:
            def handle_error(future: Future, exception: Exception):
                on_error(exception)

            executor.signals.task_failed.connect(handle_error)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Wrapper that submits the task to background executor."""

            # Inject progress callback if the function accepts it
            if on_progress:
                import inspect
                sig = inspect.signature(fn)
                if 'progress_callback' in sig.parameters:
                    kwargs['progress_callback'] = on_progress

            # Submit to executor (returns immediately)
            future = executor.submit(fn, *args, **kwargs)

            return future

        # Attach executor for manual control if needed
        wrapper.executor = executor

        return wrapper

    # Handle both @background_task and @background_task(...)
    if func is None:
        # Called with arguments: @background_task(max_workers=4)
        return decorator
    else:
        # Called without arguments: @background_task
        return decorator(func)


def cpu_task(
    func: Optional[Callable] = None,
    *,
    max_workers: Optional[int] = None,
    on_complete: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
):
    """
    Simplified decorator for CPU-intensive tasks.

    Uses ThreadExecutor by default for compatibility. For truly parallel CPU work
    (bypassing GIL), use ProcessPoolExecutor directly.

    Note: In IDA, most CPU work calls IDA SDK functions which release the GIL,
    so threads work well. Use this for pattern analysis, signature generation, etc.

    Example:
        >>> @cpu_task(on_complete=lambda r: print(f"Result: {r}"))
        ... def find_signature(data):
        ...     # CPU-intensive work here
        ...     return analyze_pattern(data)
        ...
        >>> future = find_signature(binary_data)  # Runs in background thread
    """
    return background_task(
        func,
        max_workers=max_workers,
        on_complete=on_complete,
        on_error=on_error,
        executor_type='thread'
    )


def io_task(
    func: Optional[Callable] = None,
    *,
    max_workers: Optional[int] = None,
    on_complete: Optional[Callable] = None,
    on_error: Optional[Callable] = None,
):
    """
    Simplified decorator for I/O-bound tasks.

    Alias for @background_task with executor_type='thread'.
    Use for I/O-bound work (network requests, file operations, database queries, etc.)

    Example:
        >>> @io_task(on_complete=lambda r: print(f"Downloaded: {r}"))
        ... def fetch_data(url):
        ...     return requests.get(url).json()
        ...
        >>> future = fetch_data("https://api.example.com/data")
    """
    return background_task(
        func,
        max_workers=max_workers,
        on_complete=on_complete,
        on_error=on_error,
        executor_type='thread'
    )


def parallel(max_workers: int = None):
    """
    Decorator for functions that should run multiple instances in parallel.

    Convenience decorator that sets up parallel execution with sensible defaults.
    Uses threads for compatibility - for true parallel CPU work, use ProcessPoolExecutor.

    Args:
        max_workers: Number of parallel workers (default: CPU count)

    Example:
        >>> @parallel(max_workers=8)
        ... def process_function(address):
        ...     # Analyze function at address
        ...     return analysis_result
        ...
        >>> # Process 100 functions in parallel across 8 workers
        >>> futures = [process_function(addr) for addr in function_addresses]
        >>> results = [f.result() for f in futures]
    """
    return background_task(max_workers=max_workers, executor_type='thread')


# Export all decorators
__all__ = [
    'background_task',
    'cpu_task',
    'io_task',
    'parallel',
    'get_global_executor',
]
