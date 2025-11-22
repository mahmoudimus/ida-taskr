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


def shared_memory_task(
    func: Optional[Callable] = None,
    *,
    num_chunks: int = 8,
    max_workers: Optional[int] = None,
    on_complete: Optional[Callable[[list], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
):
    """
    Decorator for processing large data using shared memory across multiple processes.

    Handles all the shared memory complexity:
    - Creates shared memory segment
    - Copies data once
    - Calculates chunk boundaries
    - Creates workers that attach to shared memory
    - Collects results
    - Cleanup (memoryview, close, unlink)

    User just writes the chunk processing logic!

    Args:
        func: The chunk processing function (auto-filled when used without parens)
        num_chunks: Number of chunks to split data into (default: 8)
        max_workers: Maximum number of parallel workers (default: num_chunks)
        on_complete: Callback when all chunks complete, receives list of results
        on_error: Callback when any chunk fails, receives exception

    The decorated function receives:
        chunk_data: memoryview of this chunk's data
        chunk_id: 0-based chunk index
        total_chunks: total number of chunks

    Returns:
        Decorated function that takes full data and returns list of all chunk results

    Example:
        >>> @shared_memory_task(num_chunks=8)
        ... def analyze_chunk(chunk_data, chunk_id, total_chunks):
        ...     # Process this chunk
        ...     signatures = find_patterns(chunk_data)
        ...     return {'chunk': chunk_id, 'sigs': signatures}
        ...
        >>> # Usage - just pass the full data!
        >>> results = analyze_chunk(binary_data)  # Returns list of 8 results
        >>> # ida-taskr handles all shared memory complexity

    Real-world IDA example:
        >>> @shared_memory_task(num_chunks=16)
        ... def find_signatures(chunk_data, chunk_id, total_chunks):
        ...     signatures = []
        ...     for i in range(len(chunk_data)):
        ...         if is_interesting_pattern(chunk_data[i:i+16]):
        ...             signatures.append(bytes(chunk_data[i:i+16]))
        ...     return signatures
        ...
        >>> # Get binary data from IDA
        >>> binary_data = ida_bytes.get_bytes(start_ea, size)
        >>> # Process in parallel - shared memory handles 8MB+ efficiently
        >>> all_signatures = find_signatures(binary_data)
    """
    import multiprocessing.shared_memory as shm_module

    def decorator(fn: Callable) -> Callable:
        """The actual decorator function."""

        @functools.wraps(fn)
        def wrapper(data: bytes) -> list:
            """
            Wrapper that handles all shared memory setup/teardown.

            Args:
                data: Full binary data to process

            Returns:
                List of results from all chunks
            """
            if not QT_AVAILABLE:
                raise RuntimeError("Qt is not available - cannot use shared memory tasks")

            # Create shared memory segment
            shm = shm_module.SharedMemory(create=True, size=len(data))

            try:
                # Copy data into shared memory (ONCE!)
                shm.buf[:len(data)] = data

                # Calculate chunk boundaries
                chunk_size = len(data) // num_chunks
                chunks_info = []

                for i in range(num_chunks):
                    start = i * chunk_size
                    # Last chunk gets any remainder
                    end = start + chunk_size if i < num_chunks - 1 else len(data)
                    chunks_info.append((start, end, i))

                # Create executor
                workers = max_workers if max_workers else num_chunks
                executor = ProcessPoolExecutor(max_workers=workers)

                try:
                    # Submit all chunks
                    futures = []
                    for start, end, chunk_id in chunks_info:
                        future = executor.submit(
                            _shared_memory_worker,
                            fn,
                            shm.name,
                            start,
                            end,
                            chunk_id,
                            num_chunks
                        )
                        futures.append(future)

                    # Collect results
                    results = []
                    for future in futures:
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            if on_error:
                                on_error(e)
                            raise

                    # Call completion callback if provided
                    if on_complete:
                        on_complete(results)

                    return results

                finally:
                    executor.shutdown(wait=True)

            finally:
                # Cleanup shared memory
                shm.close()
                shm.unlink()

        return wrapper

    # Handle both @shared_memory_task and @shared_memory_task(...)
    if func is None:
        # Called with arguments: @shared_memory_task(num_chunks=16)
        return decorator
    else:
        # Called without arguments: @shared_memory_task
        return decorator(func)


def _shared_memory_worker(fn: Callable, shm_name: str, start: int, end: int, chunk_id: int, total_chunks: int):
    """
    Worker function that attaches to shared memory and processes a chunk.

    This needs to be a module-level function so it can be pickled for ProcessPoolExecutor.

    Args:
        fn: The user's chunk processing function
        shm_name: Name of the shared memory segment
        start: Start offset in shared memory
        end: End offset in shared memory
        chunk_id: This chunk's ID
        total_chunks: Total number of chunks

    Returns:
        Result from user's chunk processing function
    """
    import multiprocessing.shared_memory as shm_module

    # Attach to existing shared memory
    shm = shm_module.SharedMemory(name=shm_name)

    try:
        # Get view of this chunk (no data copying!)
        chunk_data = memoryview(shm.buf)[start:end]

        # Call user's function with chunk data
        result = fn(chunk_data, chunk_id, total_chunks)

        return result

    finally:
        # CRITICAL: Delete memoryview before closing shared memory
        del chunk_data
        shm.close()


# Export all decorators
__all__ = [
    'background_task',
    'cpu_task',
    'io_task',
    'parallel',
    'shared_memory_task',
    'get_global_executor',
]
