"""Worker entry point for anti-deobfuscation tasks."""

import argparse
import asyncio
import concurrent.futures
import dataclasses
import multiprocessing.shared_memory
import pathlib
import sys
import threading
import time

from anti_deob.deobfuscator import MAX_PATTERN_LEN, analyze_chain, stage1_find_patterns

from ida_taskr import (
    ConnectionContext,
    DataProcessorCore,
    MessageEmitter,
    WorkerBase,
    get_logger,
)
from ida_taskr.utils import (
    AsyncEventEmitter,
    IntervalSet,
    PatchManager,
    log_execution_time,
    make_chunks,
    resolve_overlaps,
    shm_buffer,
)

logger = get_logger(__name__)

# Path to this worker script (for self-reference)
WORKER_SCRIPT_PATH = pathlib.Path(__file__)


def process_chunk(args):
    """
    Entire 4-stage pipeline over one overlapping chunk.
    Returns only those chains whose start is in the chunk's core region.
    """
    shm_name, padded_start, padded_end, core_start, core_end, base_ea, is_64 = args

    # attach shared memory
    with shm_buffer(shm_name) as shm:
        # zero-copy view of the chunk
        full_buf_mv = memoryview(shm.buf)[padded_start:padded_end]  # type: ignore
        try:
            core_valid = []
            # — Stage 1
            s1_chains = stage1_find_patterns(full_buf_mv, base_ea + padded_start)

            # ─── TRACEPOINT: Stage 1 ───────────────────────────────────
            TARGET_EA = 0x140005131
            for chain in s1_chains:
                s = chain.overall_start()
                e = s + chain.overall_length()
                if s <= TARGET_EA < e:
                    logger.warning("[TRACE][Stage1] covers 0x%X: %s", TARGET_EA, chain)
                    for seg in chain.segments:
                        abs_s = chain.base_address + seg.start
                        logger.warning(
                            "    seg @0x%X len=%d desc=%s",
                            abs_s,
                            seg.length,
                            seg.description,
                        )
                    break
            # ────────────────────────────────────────────────────────────

            for chain in s1_chains:
                # IMPORTANT: buf starts at padded_start, so bump base_ea accordingly
                ranges = analyze_chain(
                    chain, full_buf_mv, base_ea + padded_start, is_64
                )
                core_valid.extend(ranges)
            return core_valid
        finally:
            del full_buf_mv


@dataclasses.dataclass
class AsyncDeobfuscator(AsyncEventEmitter):
    shm_name: str
    data_size: int
    start_ea: int
    is_64bit: bool
    max_workers: int = 0
    executor: concurrent.futures.Executor | None = None

    def __post_init__(self):
        super().__post_init__()
        self.pause_evt = asyncio.Event()
        self.stop_evt = asyncio.Event()
        self.max_workers = self.max_workers or max(1, multiprocessing.cpu_count())
        ctx = multiprocessing.get_context("spawn")
        self.executor = self.executor or concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx
        )
        logger.info(f"executor pool created with {self.max_workers} workers")

    @log_execution_time
    async def run(self):
        await self.emit("run_started")

        # 1) define exactly max_workers chunks over the shared buffer
        buf_len = self.data_size
        chunks = list(
            make_chunks(
                buf_len,
                self.max_workers,
                max_pat=MAX_PATTERN_LEN * 2,
            )
        )
        logger.debug(
            "Splitting buffer of %d bytes into %d chunks:", buf_len, len(chunks)
        )
        for idx, (ps, pe, cs, ce) in enumerate(chunks):
            logger.info(
                "  chunk %2d: core=[0x%X..0x%X) padded=[0x%X..0x%X)",
                idx,
                cs + self.start_ea,
                ce + self.start_ea,
                ps + self.start_ea,
                pe + self.start_ea,
            )

        # 2) fire one full-pipeline task per chunk
        loop = asyncio.get_running_loop()
        jobs = [
            (
                self.shm_name,
                padded_start,
                padded_end,
                core_start,
                core_end,
                self.start_ea,
                self.is_64bit,
            )
            for padded_start, padded_end, core_start, core_end in chunks
        ]
        futures = [
            loop.run_in_executor(self.executor, process_chunk, job) for job in jobs
        ]

        # 3) wait, flatten, resolve overlaps globally
        per_chunk = await asyncio.gather(*futures)
        all_ranges = [r for grp in per_chunk for r in grp]
        final: IntervalSet = resolve_overlaps(all_ranges)

        await self.emit("run_finished", final)
        return final

    async def shutdown(self):
        self.stop_evt.set()
        if self.executor:
            self.executor.shutdown(wait=True)
        await self.emit("stopped")
        self.logger.info("AsyncDeobfuscator shutdown complete.")


class AntiDeobWorker(WorkerBase):
    """Worker implementation for anti-deobfuscation tasks."""

    def __init__(self, shm_name: str, data_size: int, start_ea: int, is_64bit: bool):
        # Pass AsyncDeobfuscator class and its arguments to WorkerBase
        emitter_args = {
            "shm_name": shm_name,
            "data_size": data_size,
            "start_ea": start_ea,
            "is_64bit": is_64bit,
            # process_chunk_fn will be implicitly passed if AsyncDeobfuscator expects it
        }
        # Note: process_chunk is used by AsyncDeobfuscator's run method,
        # so it doesn't need to be passed to WorkerBase directly here if AsyncDeobfuscator handles it.
        # If AsyncDeobfuscator needed process_chunk to be injected, that would be part of its own init.
        super().__init__(
            async_emitter_class=AsyncDeobfuscator, emitter_args=emitter_args
        )

        # Specific attributes for AntiDeobWorker, if any, can be initialized here.
        # For this example, the core logic is within AsyncDeobfuscator.
        self.logger.info("AntiDeobWorker initialized with new WorkerBase structure.")

    # The `setup` method in WorkerBase now initializes the emitter.
    # We can override `setup_custom_event_handlers` if we need specific event handling
    # beyond what WorkerBase provides by default.

    # The `process` method is now largely handled by WorkerBase.
    # We might not need to override it unless there's very specific pre/post command loop logic.

    # `handle_command` can be extended if new custom commands are needed beyond
    # what WorkerBase handles (start, stop, pause, resume, ping, set_log_level).
    # For this example, AsyncDeobfuscator doesn't introduce new commands that
    # WorkerBase needs to be aware of at this level. The existing commands
    # control the WorkerController managed by WorkerBase.

    # `cleanup` is also handled by WorkerBase to shut down the emitter.

    # Example of custom event handler setup, if needed:
    # def setup_custom_event_handlers(self):
    #     super().setup_custom_event_handlers() # Good practice if base class might add some
    #     if self.emitter_instance: # emitter_instance is set up in WorkerBase.setup()
    #         @self.emitter_instance.on("some_custom_event_from_async_deobfuscator")
    #         def on_my_custom_event(data):
    #             self.logger.info(f"Received custom event with data: {data}")
    #             if self.conn:
    #                 self.conn.send_message("custom_progress", data, status="custom_status")

    # If the results formatting in WorkerBase's send_results is not sufficient,
    # it can be overridden here. The default handles IntervalSet and lists.
    # def send_results(self, connection: ConnectionContext, results_data):
    #    self.logger.info("AntiDeobWorker formatting results...")
    #    # Custom formatting logic here
    #    super().send_results(connection, formatted_results_data)


def create_anti_deob_message_emitter(dry_run: bool = False) -> MessageEmitter:
    """Create a message emitter for anti-deobfuscation results.

    Args:
        dry_run: If True, patches will be simulated but not applied

    Returns:
        MessageEmitter configured with anti-deob event handlers
    """

    emitter = MessageEmitter()

    @emitter.on("worker_connected")
    def on_worker_connected():
        logger.info("Worker connected")

    @emitter.on("worker_message")
    def on_worker_message(message: dict):
        logger.info("Worker message: %s", message)

    @emitter.on("worker_error")
    def on_worker_error(error: str):
        logger.error("Worker error: %s", error)

    @emitter.on("worker_disconnected")
    def on_worker_disconnected():
        logger.info("Worker disconnected")

    @emitter.on("worker_results")
    def on_worker_results(results: dict):
        patch_manager = PatchManager(dry_run=dry_run)
        if results["status"] == "success":
            NOP = b"\x90"
            for patch_instructions in results["results"]:
                patch_manager.add_patch(
                    patch_instructions["address"],
                    patch_instructions["length"] * NOP,
                )
            patch_manager.apply_all()
        else:
            logger.error("Worker reported an error in results")

    return emitter


class Taskr:
    """
    Singleton wrapper for the DataProcessor instance.

    Ensures only one DataProcessor is created and shared throughout the plugin's lifetime.

    Usage:
        >>> t1 = Taskr()
        >>> t2 = Taskr()
        >>> t1 is t2
        True
        >>> t1.get() is t2.get()
        True

    The .get() method returns the singleton DataProcessor instance.
    """

    _instance = None
    _processor = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Not thread-safe, but sufficient for plugin/IDA context
            cls._instance = super().__new__(cls)
            logger.info("Initializing DataProcessor")
            # Create patch manager and message emitter for singleton
            message_emitter = create_anti_deob_message_emitter(dry_run=False)
            cls._processor = DataProcessorCore(message_emitter)
        return cls._instance

    def get(self):
        """
        Returns the singleton DataProcessor instance.

        >>> t1 = Taskr()
        >>> t2 = Taskr()
        >>> t1.get() is t2.get()
        True
        """
        return self._processor

    def pause(self):
        self.get().proc.send_command({"command": "pause"})  # type: ignore

    def resume(self):
        self.get().proc.send_command({"command": "resume"})  # type: ignore

    def stop(self):
        self.get().proc.send_command({"command": "stop"})  # type: ignore

    def start(self):
        self.get().proc.send_command({"command": "start"})  # type: ignore

    def terminate(self):
        self.get().terminate()  # type: ignore

    def ping(self):
        self.get().proc.send_command({"command": "ping"})  # type: ignore


def main():
    """Worker entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm_name", required=True)
    parser.add_argument("--data_size", type=int, required=True)
    parser.add_argument("--start_ea", type=lambda x: int(x, 0), required=True)
    parser.add_argument("--is64", type=int, default=1)
    parser.add_argument("--address", required=True)
    parser.add_argument("--authkey", required=True)

    args = parser.parse_args()
    worker = None
    try:
        # Create worker instance
        worker = AntiDeobWorker(
            shm_name=args.shm_name,
            data_size=args.data_size,
            start_ea=args.start_ea,
            is_64bit=bool(args.is64),
        )

        # WorkerBase's setup will initialize AsyncDeobfuscator
        worker.setup()  # This is crucial

        # Process with connection
        # WorkerBase's process method now handles the command loop
        with ConnectionContext(args.address, args.authkey) as conn:
            worker.process(conn)  # This starts the command loop in WorkerBase

    except Exception as e:
        logger.error("Unhandled exception in worker_main: %s", e, exc_info=True)
    finally:
        if worker:
            # WorkerBase's cleanup should handle shutting down the emitter
            # It's an async method, so needs to be run in an event loop
            try:
                asyncio.run(worker.cleanup())
            except Exception as e:
                logger.error("Exception during worker cleanup: %s", e, exc_info=True)
        logger.info("Worker main finished.")


if __name__ == "__main__":
    # If run with worker arguments, run as worker
    if len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        main()
    else:
        # If run directly, run as IDA script
        print("Running Taskr().get().run(*DataProcessorCore.get_section_data('.text'))")
        data_ea, data_bytes = DataProcessorCore.get_section_data(".text")
        if data_ea and data_bytes:
            processor = Taskr().get()
            if processor:
                processor.run(data_ea, data_bytes, str(WORKER_SCRIPT_PATH))
