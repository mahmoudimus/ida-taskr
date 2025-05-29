"""Worker entry point for anti-deobfuscation tasks."""

import argparse
import asyncio
import multiprocessing.shared_memory
import pathlib
import sys
import threading
import time

from ida_taskr import (
    ConnectionContext,
    DataProcessorCore,
    MessageEmitter,
    WorkerBase,
    get_logger,
)
from ida_taskr.utils import PatchManager

from .deobfuscator import AsyncDeobfuscator, WorkerController

logger = get_logger(__name__)

# Path to this worker script (for self-reference)
WORKER_SCRIPT_PATH = pathlib.Path(__file__)


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


class AntiDeobWorker(WorkerBase):
    """Worker implementation for anti-deobfuscation tasks."""

    def __init__(self, shm_name: str, data_size: int, start_ea: int, is_64bit: bool):
        super().__init__()
        self.shm_name = shm_name
        self.data_size = data_size
        self.start_ea = start_ea
        self.is_64bit = is_64bit

    @property
    def deob(self) -> AsyncDeobfuscator:
        return self._deob

    @deob.setter
    def deob(self, value: AsyncDeobfuscator):
        self._deob = value

    @property
    def conn(self) -> ConnectionContext | None:
        return self._conn

    @conn.setter
    def conn(self, value: ConnectionContext | None):
        self._conn = value

    def setup(self, **kwargs):
        """Initialize the deobfuscator."""
        self.deob = AsyncDeobfuscator(
            shm_name=self.shm_name,
            data_size=self.data_size,
            start_ea=self.start_ea,
            is_64bit=self.is_64bit,
        )

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up progress tracking and event handlers."""

        @self.deob.on("run_started")
        def on_run_started():
            self.logger.info("▶️  Pipeline starting")
            if self.conn:
                self.conn.send_message(
                    "progress", 0.0, status="running", stage="starting"
                )

        @self.deob.on("run_finished")
        def on_run_finished(ch):
            self.logger.info("✅ Processed %d chunks", len(ch))
            if self.conn:
                self.conn.send_message(
                    "progress",
                    0.95,
                    status="finalizing",
                    stage="stage4_complete",
                    chunks_count=len(ch),
                )

        @self.deob.on("stopped")
        def on_stopped():
            self.logger.info("🛑 Worker shutting down")
            if self.conn:
                self.conn.send_message("status", "stopped", status="stopped")

    def process(self, connection: ConnectionContext, **kwargs):
        """Main processing loop."""
        self.conn = connection  # Store for event handlers

        # Set up controller
        self.controller = WorkerController(self.deob)

        # Send initial ready message
        connection.send_message("status", "connected", status="ready")

        self.logger.info("Starting command loop...")

        try:
            while True:
                try:
                    if not connection.closed and not connection.poll(timeout=0.5):
                        continue
                    cmd = connection.recv()
                    self.logger.debug(f"← Received command: {cmd}")
                except EOFError:
                    self.logger.error("Connection closed by parent")
                    break

                # Handle command
                if isinstance(cmd, dict):
                    if not self.handle_command(cmd, connection):
                        break

                else:
                    self.logger.warning(
                        f"Received unexpected command type: {type(cmd)}"
                    )
                    connection.send_message(
                        "error",
                        f"Expected dict command, got {type(cmd)}",
                        status="error",
                    )
            # Ensure pipeline stops
            if self.controller:
                self.controller.stop()

        finally:
            self.conn = None  # Clear reference

        # Wait for results
        self.logger.info("Waiting for pipeline to finish...")
        results = self.controller.join() if self.controller else None

        if results:
            # Format and send results
            asjson = [
                {
                    "address": s,
                    "length": e - s,
                    "end": e,
                }
                for s, e in results.as_tuples()
            ]

            self.logger.info(f"Sending {len(asjson)} results...")
            connection.send_message(
                "status", "sending_results", status="sending_results"
            )
            time.sleep(5)  # Give IDA time to prepare
            connection.send_message(
                "result", asjson, status="success", count=len(asjson)
            )
            connection.send_message("status", "results_sent", status="results_sent")

    def handle_command(self, cmd: dict, connection: ConnectionContext) -> bool:
        """Handle custom commands."""
        cmd_type = cmd.get("command")
        if cmd_type in ["stop", "exit", "shutdown"]:
            self.logger.info("Received exit command")
            return False
        elif cmd_type == "ping":
            connection.send_message("status", "pong", status="running")
        elif cmd_type == "pause":
            self.controller.pause()
            connection.send_message("status", "paused", status="paused")
        elif cmd_type == "resume":
            self.controller.resume()
            connection.send_message("status", "resumed", status="running")
        elif cmd_type == "start":
            self.controller.start()
            connection.send_message("status", "started", status="running")
        elif cmd_type == "set_log_level":
            level = cmd.get("level")
            if level is None:
                self.logger.error("Log level is required")
                connection.send_message(
                    "error", "Log level is required", status="error"
                )
                return True
            self.logger.setLevel(level)
            # self.controller.set_log_level(level)
            logger.info(f"Worker log level set to {level}")
            connection.send_message(
                "status",
                f"log_level_set:{level}",
                status="running",
            )
        else:
            self.logger.warning(f"Unknown command type: {cmd_type}")
            connection.send_message(
                "error", f"Unknown command: {cmd_type}", status="error"
            )
        return True

    async def cleanup(self):
        """Clean up resources."""
        if self.controller:
            self.controller.stop()
        if self.deob:
            await self.deob.shutdown()


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
        # Create worker
        worker = AntiDeobWorker(
            shm_name=args.shm_name,
            data_size=args.data_size,
            start_ea=args.start_ea,
            is_64bit=bool(args.is64),
        )

        # Set up worker
        worker.setup()

        # Process with connection
        with ConnectionContext(args.address, args.authkey) as conn:
            worker.process(conn)

    except Exception as e:
        logger.error(f"Unhandled exception in worker: {e}", exc_info=True)
    finally:
        if worker:
            asyncio.run(worker.cleanup())
        logger.info("Worker finished.")


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
