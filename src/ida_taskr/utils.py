"""Utility functions for the anti-deobfuscation plugin."""

import asyncio
import atexit
import collections
import contextlib
import dataclasses
import enum
import functools
import logging
import multiprocessing
import multiprocessing.shared_memory
import pathlib
import time
import typing
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from typing import Any, Callable, Generic, TypeVar, overload

from .helpers import get_logger

logger = get_logger(__name__)


def humanize_bytes(
    num_bytes: int, precision: int = 2, units: list[str] = ["B", "KB", "MB", "GB"]
) -> str:
    """
    Convert a byte count into a human-friendly string with units.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    if num_bytes == 0:
        return "0 B"
    idx = 0
    value = float(num_bytes)
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if value.is_integer():
        return f"{int(value)} {units[idx]}"
    else:
        return f"{value:.{precision}f} {units[idx]}"


class DataProcessorCore:
    """Core processor for managing deobfuscation tasks."""

    _shared_memory = None

    def __init__(self, message_emitter):
        """Initialize the DataProcessorCore.

        Args:
            message_emitter: MessageEmitter instance to handle worker communication
        """
        from .protocols import MessageEmitter

        if not isinstance(message_emitter, MessageEmitter):
            raise TypeError("message_emitter must be a MessageEmitter instance")

        self.message_emitter = message_emitter
        self.proc = None
        atexit.register(self.terminate)

    @staticmethod
    def get_section_data(
        section_name: str,
        max_size: int = 120 * 1024 * 1024,
        min_size: int = 1024,
    ) -> tuple[int, bytes]:
        """Get the data of a section by name."""
        try:
            import ida_bytes
            import ida_segment
            import idaapi
        except ImportError:
            logger.error("IDA Pro modules not available")
            return 0, b""  # Use 0 instead of idaapi.BADADDR when not available

        seg = ida_segment.get_segm_by_name(section_name)
        if not seg:
            logger.error("Section %s not found", section_name)
            return idaapi.BADADDR, b""

        data_ea = seg.start_ea
        data_size = seg.end_ea - seg.start_ea

        if data_size > max_size:
            data_size = max_size
            logger.warning(
                "Limiting section data size to %s", humanize_bytes(data_size)
            )
        elif data_size < min_size:
            logger.error(
                "%s section is too small (%s)", section_name, humanize_bytes(data_size)
            )
            return idaapi.BADADDR, b""

        logger.info(
            "Reading %s from address %s", humanize_bytes(data_size), hex(data_ea)
        )

        data_bytes = ida_bytes.get_bytes(data_ea, data_size)
        if not data_bytes or len(data_bytes) != data_size:
            logger.error("Failed to read section data")
            return idaapi.BADADDR, b""

        return data_ea, data_bytes

    @staticmethod
    def from_range(start_ea: int, end_ea: int):
        """Get the data of a section by name and return the start address and the bytes."""
        try:
            import ida_bytes
        except ImportError:
            logger.error("IDA Pro modules not available")
            return 0, b""

        data_bytes = ida_bytes.get_bytes(start_ea, end_ea - start_ea)
        return start_ea, data_bytes

    def run(
        self, start_ea: int, bytes_to_process: bytes, worker_script_path: str, **kwargs
    ):
        """Run the deobfuscation process.

        Args:
            start_ea: Starting address for processing
            bytes_to_process: Binary data to process
            worker_script_path: Path to the worker script
            **kwargs: Additional arguments passed to worker
        """
        from .launcher import WorkerLauncher

        data_size = len(bytes_to_process)

        # Create shared memory
        self._shared_memory = multiprocessing.shared_memory.SharedMemory(
            create=True, size=data_size
        )
        self._shared_memory.buf[:data_size] = bytes_to_process

        # Launch worker with provided message emitter
        self.proc = WorkerLauncher(self.message_emitter)

        worker_args = {
            "shm_name": self._shared_memory.name,
            "data_size": data_size,
            "start_ea": hex(start_ea),
            **kwargs,
        }

        # Add IDA bitness if available
        try:
            import ida_ida

            worker_args["is64"] = "1" if ida_ida.inf_is_64bit() else "0"
        except ImportError:
            logger.warning("IDA Pro modules not available, bitness detection skipped")

        if not self.proc.launch_worker(str(worker_script_path), worker_args):
            self.terminate()
            logger.error("Failed to start worker process")
            return

    def terminate(self):
        """Terminate and clean up."""
        logger.info("Terminating...")
        if self.proc and not self.proc.is_not_running():
            self.proc.stop_worker()
            self.proc = None
        self._cleanup_shared_memory()
        logger.info("Terminated.")

    def _cleanup_shared_memory(self):
        """Clean up shared memory."""
        if not self._shared_memory:
            return
        try:
            self._shared_memory.close()
            shm = multiprocessing.shared_memory.SharedMemory(self._shared_memory.name)
            shm.unlink()
            logger.info("Shared memory unlinked: %s", self._shared_memory.name)
        except FileNotFoundError:
            logger.warning(
                "Shared memory already unlinked: %s", self._shared_memory.name
            )
        except PermissionError as e:
            logger.error("Permission error unlinking shared memory: %s", e)
        except Exception as e:
            logger.error(
                "Unexpected error unlinking shared memory: %s", e, exc_info=True
            )
        finally:
            self._shared_memory = None


class emit:
    def __init__(self, event):
        self.event = event

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(inst, *args, **kwargs):
            result = fn(inst, *args, **kwargs)
            inst.emit(self.event)
            return result

        return wrapper


T = TypeVar("T")


class reify(Generic[T]):
    """
    Acts similar to a property, except the result will be
    set as an attribute on the instance instead of recomputed
    each access.
    """

    def __init__(self, fn: Callable[..., T]) -> None:
        self.fn = fn
        # Copy function attributes to preserve metadata
        self.__name__ = getattr(fn, "__name__", "<unknown>")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", "") or ""
        self.__qualname__ = getattr(fn, "__qualname__", "") or ""
        self.__annotations__ = getattr(fn, "__annotations__", {})

    @overload
    def __get__(self, instance: None, owner: type) -> "reify[T]": ...

    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...

    def __get__(self, instance: Any, owner: type) -> "T | reify[T]":
        if instance is None:
            return self

        fn = self.fn
        val = fn(instance)
        setattr(instance, fn.__name__, val)
        return val


class EventEmitter:
    @reify
    def _listeners(self):
        return collections.defaultdict(set)

    def on(self, event, handler=None):
        """Register an event handler for the given event."""
        if handler:
            self._listeners[event].add(handler)
            return handler

        @functools.wraps(self.on)
        def decorator(func):
            self.on(event, func)
            return func

        return decorator

    def once(self, event, handler):
        @functools.wraps(handler)
        def once_handler(*args, **kwargs):
            self.remove(event, once_handler)
            return handler(*args, **kwargs)

        self.on(event, once_handler)

    def remove(self, event, handler):
        self._listeners[event].discard(handler)

    def emit(self, event, *args, **kwargs):
        for handler in list(self._listeners[event]):
            handler(*args, **kwargs)


@dataclasses.dataclass
class AsyncEventEmitter(ABC):
    def __post_init__(self):
        self._listeners = collections.defaultdict(set)
        self.pause_evt = asyncio.Event()
        self.stop_evt = asyncio.Event()
        self.logger = get_logger(self.__class__.__name__)

    def on(self, event, handler=None):
        """Register an event handler for the given event."""
        if handler:
            self._listeners[event].add(handler)
            return handler

        @functools.wraps(self.on)
        def decorator(func):
            self.on(event, func)
            return func

        return decorator

    def once(self, event, handler):
        @functools.wraps(handler)
        def once_handler(*args, **kwargs):
            self.remove(event, once_handler)
            return handler(*args, **kwargs)

        self.on(event, once_handler)

    def remove(self, event, handler):
        self._listeners[event].discard(handler)

    async def emit(self, event, *args):
        for handler in list(self._listeners[event]):
            await handler(*args)

    @abstractmethod
    async def run(self):
        """Core asynchronous task execution logic."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Cleanup resources for the async task."""
        pass


def log_execution_time(func, loglvl=logging.INFO):
    """
    Decorator to log the execution time of async stage methods.

    >>> import asyncio, logging
    >>> logging.basicConfig(level=logging.INFO)
    >>> class Dummy:
    ...     @log_execution_time
    ...     async def foo(self):
    ...         await asyncio.sleep(0.01)
    ...         return 42
    >>> d = Dummy()
    >>> asyncio.run(d.foo())
    42
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.log(loglvl, f"{func.__qualname__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


@dataclasses.dataclass
class Range:
    """A range of addresses with a start (inclusive) and end (exclusive)."""

    start: int
    end: int

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start must be less than end")

    def __contains__(self, addr: int) -> bool:
        """Check if an address is within this range."""
        return self.start <= addr < self.end

    def __len__(self) -> int:
        """Return the size of the range in bytes."""
        return self.end - self.start

    def overlaps(self, other: "Range") -> bool:
        """Check if this range overlaps with another range."""
        return self.start < other.end and other.start < self.end

    def merge(self, other: "Range") -> "Range":
        return Range(min(self.start, other.start), max(self.end, other.end))


class IntervalSet:
    """
    Sorted, non-overlapping list of Range objects with O(log n) insertion.
    """

    __slots__ = ("_ranges",)

    def __init__(self) -> None:
        self._ranges: list[Range] = []

    def __iter__(self):
        return iter(self._ranges)

    def __len__(self):
        return len(self._ranges)

    # --- public ------------------------------------------------------------
    def add(self, new: Range) -> None:
        """
        Insert `new` and coalesce any overlaps / adjacencies in-place.
        """
        # Fast-path: first interval
        if not self._ranges:
            self._ranges.append(new)
            return

        # Binary-search insertion point by *start*
        idx = bisect_left(
            self._ranges, new.start, key=lambda r: r.start
        )  # Python 3.10+

        # Extend backward if necessary
        if idx > 0 and self._ranges[idx - 1].end >= new.start:
            idx -= 1

        # Merge forward while overlapping
        while idx < len(self._ranges) and new.overlaps(self._ranges[idx]):
            new = new.merge(self._ranges[idx])
            del self._ranges[idx]

        # Also coalesce "touching" intervals (…,end==new.start or vice-versa)
        if idx < len(self._ranges) and new.end == self._ranges[idx].start:
            new = new.merge(self._ranges[idx])
            del self._ranges[idx]
        if idx > 0 and self._ranges[idx - 1].end == new.start:
            new = new.merge(self._ranges[idx - 1])
            del self._ranges[idx - 1]
            idx -= 1

        self._ranges.insert(idx, new)

    # ­— optional helpers ---------------------------------------------------
    def covers(self, addr: int) -> bool:
        i = bisect_right(self._ranges, addr, key=lambda r: r.start) - 1
        return i >= 0 and addr < self._ranges[i].end

    def as_tuples(self):
        return [(r.start, r.end) for r in self._ranges]


def resolve_overlaps(ranges: list[Range]) -> IntervalSet:
    """
    Fast, linear-time overlap resolution: keep only the first chain
    whose start is ≥ the furthest end so far.
    """
    logger.info(f"Resolving overlaps among {len(ranges)} ranges")
    intervals = IntervalSet()

    for r in ranges:
        intervals.add(r)

        # decide whether to keep the chain object itself
        last_end = intervals.as_tuples()[-1][1]  # rightmost byte so far
        target = r.end
        if target == last_end:  # this chain extended the interval set
            logger.info(f"  Accepted (or widened): {r.start:X}-{r.end:X}")
        else:
            logger.info(f"  Rejected overlap: {r.start:X}-{r.end:X}")

    return intervals


class PatchManager:
    """Manages deferred patch operations."""

    class Mode(enum.Enum):
        PATCH = enum.auto()  # Use ida_bytes.patch_bytes
        PUT = enum.auto()  # Use ida_bytes.put_bytes

    def __init__(
        self,
        patch_mode: Mode = Mode.PATCH,
        dry_run: bool = False,
        auto_clear: bool = True,
    ):
        self.dry_run = dry_run
        self.patch_mode = patch_mode
        self.pending_patches: list[DeferredPatchOp] = []
        self.auto_clear = auto_clear
        logger.info(
            "PatchManager initialized (dry_run=%s, mode=%s)",
            self.dry_run,
            self.patch_mode.name,
        )

    def add_patch(self, address: int, byte_values: bytes):
        """Creates and queues a DeferredPatchOp."""
        op = DeferredPatchOp(address, byte_values, self.patch_mode)
        self.pending_patches.append(op)
        logger.debug("Queued patch operation: %s", op)

    def apply_all(self, dry_run_override: bool | None = None) -> bool:
        """Applies all queued patch operations."""
        logger.info("Applying %d queued patches...", len(self))
        success_count = 0
        fail_count = 0

        if dry_run_override is None:
            # None is a sentinel value here that represents "use the default"
            dry_run_override = self.dry_run

        for op in self.pending_patches:
            if op.apply(dry_run_override):
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            "Patch application complete. Success: %d, Failed: %d",
            success_count,
            fail_count,
        )
        if self.auto_clear:
            self.pending_patches.clear()  # Clear the list after applying
        return fail_count == 0  # Return True if all patches were applied successfully

    def __len__(self) -> int:
        return len(self.pending_patches)


@dataclasses.dataclass(repr=False)
class DeferredPatchOp:
    """Class to store patch operations that will be applied later."""

    address: int
    byte_values: bytes
    mode: PatchManager.Mode
    dry_run: bool = False

    @classmethod
    def patch(cls, address: int, byte_values: bytes, dry_run: bool = False):
        return cls(address, byte_values, PatchManager.Mode.PATCH, dry_run)

    @classmethod
    def put(cls, address: int, byte_values: bytes, dry_run: bool = False):
        return cls(address, byte_values, PatchManager.Mode.PUT, dry_run)

    def apply(self, dry_run_override: bool = False) -> bool:
        """Apply the patch operation using either patch_bytes or put_bytes based on mode."""

        is_dry_run = dry_run_override or self.dry_run
        logger.info(
            "[*] %sPatching decrypted chunk %s at 0x%X (size: %d)",
            "(Dry Run) " if is_dry_run else "",
            ("revertably" if self.mode == PatchManager.Mode.PATCH else "destructively"),
            self.address,
            len(self.byte_values),
        )
        success = True
        if is_dry_run:
            return success

        try:
            import idaapi  # todo: make this a protocol + pass to the manager

            func = (
                idaapi.put_bytes
                if self.mode == PatchManager.Mode.PUT
                else idaapi.patch_bytes
            )
            func(self.address, self.byte_values)
        except Exception as e:
            logger.error(f"Failed to apply patch {self}: {e}", exc_info=True)
            success = False
        return success

    def __str__(self):
        """String representation with hex formatting."""
        dry_run_str = " (dry run)" if self.dry_run else ""
        return f"{self.__class__.__name__}({len(self.byte_values)} bytes, mode={self.mode.name}{dry_run_str} @ address=0x{self.address:X})"

    __repr__ = __str__


def make_chunks(buf_len: int, n_chunks: int, max_pat: int):
    """
    Yield exactly n_chunks tuples of
      (padded_start, padded_end, core_start, core_end).

    * core ranges partition [0, buf_len) evenly by floor division.
    * padded ranges extend each core by (max_pat-1) on both sides,
      clamped to [0, buf_len].
    """
    for i in range(n_chunks):
        # uniform core split
        core_start = (buf_len * i) // n_chunks
        core_end = (buf_len * (i + 1)) // n_chunks
        core_len = core_end - core_start

        # padding
        padded_start = max(0, core_start - (max_pat - 1))
        padded_end = min(buf_len, core_end + (max_pat - 1))
        padded_len = padded_end - padded_start

        logger.debug(
            "Chunk %2d/%d: "
            "core=[%#x-%#x) (%d bytes), "
            "padded=[%#x-%#x) (%d bytes)",
            i,
            n_chunks,
            core_start,
            core_end,
            core_len,
            padded_start,
            padded_end,
            padded_len,
        )

        yield padded_start, padded_end, core_start, core_end


@contextlib.contextmanager
def shm_buffer(
    name: str, buf_len: int | None = None
) -> typing.Generator[
    multiprocessing.shared_memory.SharedMemory | memoryview, None, None
]:
    """
    context manager to access the shared memory buffer.
    if buf_len is not provided, then the buffer will be the raw shm memory
    buffer else it will be a byte slice of the shm memory buffer.

    Usage:
        with shm_buffer(name=..) as buf:
            # use buf
    """
    shm = multiprocessing.shared_memory.SharedMemory(name=name)
    try:
        yield shm.buf[:buf_len] if buf_len else shm
    finally:
        shm.close()
