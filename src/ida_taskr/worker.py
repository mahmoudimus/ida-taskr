"""Worker-side components for the task worker manager."""

import asyncio
import concurrent.futures
import math
import multiprocessing
import multiprocessing.connection
import threading
import time
import typing
import uuid

from .helpers import get_logger
from .protocols import WorkerProtocol
from .utils import AsyncEventEmitter, IntervalSet

logger = get_logger()


class ConnectionContext:
    """
    Context manager for a multiprocessing.connection.Connection.
    Ensures the connection is closed on exit.
    """

    def __init__(self, address: str, authkey: bytes | str, chunk_size: int = 1024):
        host, port_str = address.split(":")
        self.host = host
        self.port = int(port_str)

        if isinstance(authkey, str):
            authkey = bytes.fromhex(authkey)

        assert isinstance(authkey, bytes), f"Invalid authkey type: {type(authkey)}"
        self.authkey = authkey
        self._conn = None
        self.chunk_size = chunk_size

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    @property
    def conn(self) -> multiprocessing.connection.Connection:
        if self._conn is None:
            self._conn = multiprocessing.connection.Client(
                self.address, family="AF_INET", authkey=self.authkey
            )
            logger.info(f"Connected to {self.address}")
        return self._conn

    def send_message(self, msg_type: str, data, **kwargs) -> bool:
        """
        Send a structured message through the connection.

        If data is a long list, split it into chunks.
        """
        try:
            if isinstance(data, list) and len(data) > self.chunk_size:
                message_id = uuid.uuid4().hex
                total_chunks = math.ceil(len(data) / self.chunk_size)

                for idx in range(total_chunks):
                    part = data[idx * self.chunk_size : (idx + 1) * self.chunk_size]
                    msg = {
                        "type": msg_type,
                        "data": part,
                        "timestamp": time.time(),
                        "message_id": message_id,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        **kwargs,
                    }
                    self.conn.send(msg)
                logger.debug(
                    f"→ Streamed {len(data)} items in {total_chunks} chunks under id {message_id}"
                )
                return True

            # small or non-list payload: single shot
            msg = {
                "type": msg_type,
                "data": data,
                "timestamp": time.time(),
                **kwargs,
            }
            self.conn.send(msg)
            logger.debug(f"→ Sent single message: {msg_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            return False

    @property
    def closed(self):
        return self.conn.closed

    @property
    def readable(self):
        return self.conn.readable

    @property
    def writable(self):
        return self.conn.writable

    def fileno(self):
        return self.conn.fileno()

    def recv(self):
        return self.conn.recv()

    def poll(self, timeout=0.0):
        return self.conn.poll(timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PROPOGATE = False
        SUPPRESS = True
        if exc_type:
            logger.error("Connection closed by parent")
            return PROPOGATE

        if self.conn is not None:
            try:
                logger.info("Closing worker-side connection.")
                self.conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        return SUPPRESS


class WorkerController:
    """Wrap an AsyncEventEmitter in its own event loop"""

    def __init__(self, emitter_instance: AsyncEventEmitter):
        self.emitter = emitter_instance
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._result = None
        self._started = False

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self._result = self.loop.run_until_complete(self.emitter.run())
        except Exception as e:
            logger.error(f"Exception in worker thread loop: {e}", exc_info=True)
            self._result = None

    def start(self):
        if self._started:
            logger.warning("Start called on an already started worker controller.")
            return
        self._thread.start()
        self._started = True

    def pause(self):
        if not self._started:
            logger.warning("Pause called before worker controller was started.")
            return
        logger.info("▶️  Pausing...")
        self.loop.call_soon_threadsafe(self.emitter.pause_evt.set)

    def resume(self):
        if not self._started:
            logger.warning("Resume called before worker controller was started.")
            return
        logger.info("▶️  Resuming...")
        self.loop.call_soon_threadsafe(self.emitter.pause_evt.clear)

    def stop(self):
        if not self._started:
            # Even if not started, set stop event for consistency if needed
            # self.loop.call_soon_threadsafe(self.emitter.stop_evt.set) # Maybe not necessary if loop never runs
            logger.info("🛑  Stopping (controller not started)...")
            if hasattr(
                self.emitter, "stop_evt"
            ):  # Emitter might not be fully initialized
                self.loop.call_soon_threadsafe(self.emitter.stop_evt.set)
            return
        logger.info("🛑  Stopping...")
        self.loop.call_soon_threadsafe(self.emitter.stop_evt.set)

    def join(self):
        if not self._started:
            logger.warning(
                "Join called before worker controller was started. Returning current result."
            )
            return self._result

        if self._thread.is_alive():
            self._thread.join()
        else:
            logger.info(
                "Worker thread was not alive when join was called (already finished or failed?)."
            )
        self._started = False  # Reset for potential restart, though typically not done.
        return self._result

    def set_log_level(self, level):
        # This assumes the logger used by emitter.run() is the global one or accessible
        # If emitter uses its own logger, it needs a method to set its level.
        # For now, we assume it affects the logger instance used within emitter.
        # A more robust solution might involve passing a logger instance or a config.
        logger.setLevel(level)
        if hasattr(self.emitter, "logger"):
            self.emitter.logger.setLevel(level)
        logger.info(f"Log level set to {level} for controller and its emitter.")


class WorkerBase(WorkerProtocol):
    """Base class for worker implementations."""

    def __init__(
        self,
        async_emitter_class: typing.Type[AsyncEventEmitter] | None = None,
        emitter_args: dict | None = None,
        process_chunk_fn: typing.Callable | None = None,
    ):
        self.async_emitter_class = async_emitter_class
        self.emitter_args = emitter_args or {}
        self.process_chunk_fn = process_chunk_fn  # User-provided chunk processor
        self.emitter_instance: AsyncEventEmitter | None = None
        self.controller: WorkerController | None = None
        self.conn: ConnectionContext | None = None

        self._commands = {
            "stop": self._handle_stop,
            "pause": self._handle_pause,
            "resume": self._handle_resume,
            "start": self._handle_start,  # Added start command
            "ping": self._handle_ping,  # Added ping command
            "set_log_level": self._handle_set_log_level,  # Added log level command
        }
        self.logger = get_logger(self.__class__.__name__)
        self._running = False  # Tracks if the process loop is active
        # self._paused is managed by the emitter's pause_evt now

    def setup(self, **kwargs):
        """Default setup: Initializes the AsyncEventEmitter if provided."""
        if self.async_emitter_class:
            # Pass process_chunk_fn to the emitter if it accepts it
            # This requires the AsyncEventEmitter subclass to handle it in its __init__ or a setter
            current_emitter_args = self.emitter_args.copy()
            if self.process_chunk_fn:
                current_emitter_args["process_chunk_fn"] = self.process_chunk_fn

            self.emitter_instance = self.async_emitter_class(**current_emitter_args)
            self._setup_default_event_handlers()
        else:
            self.logger.warning(
                "No async_emitter_class provided for WorkerBase. Task processing will not occur."
            )

    def _setup_default_event_handlers(self):
        """Set up default event handlers for the emitter instance."""
        if not self.emitter_instance:
            return

        @self.emitter_instance.on("run_started")
        def on_run_started():
            self.logger.info("▶️  Task starting")
            if self.conn:
                self.conn.send_message(
                    "progress", 0.0, status="running", stage="starting"
                )

        @self.emitter_instance.on("run_finished")
        def on_run_finished(results):
            # The structure of 'results' depends on the emitter's run() method.
            # For AntiDeobWorker, it's an IntervalSet.
            # We need a generic way to count/summarize or pass through.
            count = (
                len(results)
                if hasattr(results, "__len__")
                else (1 if results is not None else 0)
            )
            self.logger.info("✅ Task finished, processed %s items/results.", count)
            if self.conn:
                self.conn.send_message(
                    "progress",
                    0.95,  # Assuming this is a placeholder for actual progress
                    status="finalizing",
                    stage="task_complete",  # Generic stage
                    items_count=count,
                )

        @self.emitter_instance.on("stopped")
        def on_stopped():
            self.logger.info("🛑 Emitter shutting down")
            if self.conn:
                self.conn.send_message("status", "stopped", status="stopped")

        # Allow subclasses to add more specific handlers
        self.setup_custom_event_handlers()

    def setup_custom_event_handlers(self):
        """Subclasses can override this to add their own event handlers for the emitter."""
        pass

    async def cleanup(self):
        """Default cleanup: Shuts down the emitter and controller."""
        if self.controller:
            self.controller.stop()
            # Join should ideally happen, but might block. Consider if necessary or how to handle.
            # self.controller.join() # This might be problematic in an async cleanup

        if self.emitter_instance:
            await self.emitter_instance.shutdown()
        self.logger.info("Worker cleanup complete.")

    def handle_command(self, cmd: dict, conn: ConnectionContext) -> bool:
        """
        Handle standard commands. Override for custom commands.
        Returns True to continue processing, False to exit.
        """
        cmd_type = cmd.get("command")
        if cmd_type is None:
            return True
        handler = self._commands.get(cmd_type)
        if handler:
            return handler(cmd, conn)
        return True

    def _handle_stop(self, cmd, conn):
        """Handle stop command."""
        self.logger.info("Received stop command.")
        self._running = False  # Signal process loop to stop
        if self.controller:
            self.controller.stop()
        conn.send_message("status", "stopped", status="stopped")
        return False  # Exit process loop

    def _handle_pause(self, cmd, conn):
        """Handle pause command."""
        if self.controller:
            self.controller.pause()
            conn.send_message("status", "paused", status="paused")
        else:
            conn.send_message("error", "Not started", status="error")
        return True

    def _handle_resume(self, cmd, conn):
        """Handle resume command."""
        if self.controller:
            self.controller.resume()
            conn.send_message("status", "resumed", status="running")
        else:
            conn.send_message("error", "Not started", status="error")
        return True

    def _handle_start(self, cmd, conn):
        """Handle start command."""
        if not self.emitter_instance:
            self.logger.error("Cannot start: emitter_instance not initialized.")
            conn.send_message(
                "error", "Worker not properly configured (no emitter)", status="error"
            )
            return True  # Stay in command loop

        if self.controller and self.controller._started:
            self.logger.warning("Start command received, but already started.")
            conn.send_message("status", "already_running", status="running")
            return True

        self.logger.info(
            "Received start command. Initializing and starting controller."
        )
        self.controller = WorkerController(self.emitter_instance)
        self.controller.start()
        conn.send_message("status", "started", status="running")
        return True

    def _handle_ping(self, cmd, conn):
        """Handle ping command."""
        conn.send_message(
            "status", "pong", status="running" if self._running else "idle"
        )
        return True

    def _handle_set_log_level(self, cmd, conn):
        """Handle set_log_level command."""
        level = cmd.get("level")
        if level is None:
            self.logger.error("Log level is required for set_log_level command.")
            conn.send_message("error", "Log level not specified", status="error")
            return True

        try:
            # self.logger.setLevel(level) # WorkerBase logger
            # if self.emitter_instance and hasattr(self.emitter_instance, 'logger'):
            #     self.emitter_instance.logger.setLevel(level)
            if self.controller:
                self.controller.set_log_level(level)  # Propagates to emitter logger too
            else:  # Set global logger if controller not up
                logger.setLevel(level)

            self.logger.info(
                f"Log level set to {level}"
            )  # Changed from logger.info to self.logger.info
            conn.send_message("status", f"log_level_set:{level}", status="running")
        except Exception as e:
            self.logger.error(f"Failed to set log level: {e}", exc_info=True)
            conn.send_message(
                "error", f"Failed to set log level: {str(e)}", status="error"
            )
        return True

    def process(self, connection: ConnectionContext, **kwargs):
        """Main processing loop - implement in subclasses."""
        # raise NotImplementedError("Subclasses must implement process()")
        self.conn = connection
        self._running = True

        # Send initial ready message
        connection.send_message("status", "connected", status="ready")
        self.logger.info("Worker connected, awaiting commands...")

        try:
            while self._running:
                try:
                    # Poll for messages, timeout allows checking self._running
                    if not connection.closed and not connection.poll(timeout=0.5):
                        # self.logger.debug("No message, continuing poll.")
                        if not self._running:  # Check if stop was called during poll
                            break
                        continue
                    if connection.closed:  # check before recv
                        self.logger.error(
                            "Connection closed by parent (detected before recv)."
                        )
                        self._running = False
                        break
                    cmd = connection.recv()
                    self.logger.debug(f"← Received command: {cmd}")
                except EOFError:
                    self.logger.error("Connection closed by parent (EOFError).")
                    self._running = False  # Ensure loop termination
                    break
                except (ConnectionResetError, BrokenPipeError) as e:
                    self.logger.error(f"Connection error: {e}")
                    self._running = False  # Ensure loop termination
                    break
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error receiving command: {e}", exc_info=True
                    )
                    # Optionally send error message back if connection is still usable
                    # connection.send_message("error", f"Error receiving command: {str(e)}", status="error")
                    # Depending on error, might need to stop.
                    # For now, continue to allow graceful shutdown via 'stop' command.
                    continue

                if (
                    not self._running
                ):  # check after recv, if stop was part of the message processing
                    break

                # Handle command
                if isinstance(cmd, dict):
                    if not self.handle_command(cmd, connection):
                        self._running = False  # Command handler requested exit
                        break
                elif cmd is None:  # Typically means connection closed
                    self.logger.info("Received None, likely connection closed.")
                    self._running = False
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
            # Loop exited, ensure controller stops if it was started
            if self.controller:
                self.logger.info(
                    "Process loop finished. Ensuring controller is stopped."
                )
                self.controller.stop()

        finally:
            self.logger.info("Process loop finalizing.")
            if self.controller:
                self.logger.info("Waiting for task controller to join...")
                results = self.controller.join()
                if results is not None:
                    self.send_results(connection, results)
                else:
                    self.logger.info(
                        "No results from controller or task did not complete successfully."
                    )
            else:
                self.logger.info("Controller was not active or not used.")

            self.conn = None  # Clear connection reference
            self.logger.info("Worker process loop ended.")

    def send_results(self, connection: ConnectionContext, results_data):
        """Formats and sends results. Subclasses can override for custom formatting."""
        # Default: assume results_data is list-like or an IntervalSet
        if isinstance(results_data, IntervalSet):
            as_json = [
                {"address": s, "length": e - s, "end": e}
                for s, e in results_data.as_tuples()
            ]
        elif isinstance(results_data, list):
            # Assume it's already in a suitable JSON-serializable list format
            as_json = results_data
        elif results_data is not None:
            # Attempt to make it a list if it's a single non-list item
            as_json = [results_data]
        else:
            as_json = []

        if as_json:
            self.logger.info(f"Sending {len(as_json)} results...")
            connection.send_message(
                "status", "sending_results", status="sending_results"
            )
            # Consider making sleep configurable or removing if IDA side handles UI updates better
            # time.sleep(1) # Shorter sleep or make IDA side robust
            connection.send_message(
                "result", as_json, status="success", count=len(as_json)
            )
            connection.send_message("status", "results_sent", status="results_sent")
        else:
            self.logger.info("No results to send or results were empty.")
            # Send a success message even if results are empty, or a specific no_results status
            connection.send_message(
                "result", [], status="success", count=0, note="No results generated"
            )

    # Subclasses might need to implement their own run_async_task or similar
    # if the AsyncEventEmitter pattern is not sufficient.
