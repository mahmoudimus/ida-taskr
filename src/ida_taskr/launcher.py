"""IDA-side components for launching and managing worker processes."""

import multiprocessing
import multiprocessing.connection
import os
import pickle
import select
import sys

from .helpers import MultiprocessingHelper, get_logger
from .protocols import MessageEmitter
from .qt_compat import QtCore, Signal, QProcessEnvironment, QT_AVAILABLE

logger = get_logger()


class TemporarilyDisableNotifier:
    """Context manager to temporarily disable a QSocketNotifier."""

    def __init__(self, notifier):
        self.notifier = notifier
        self.was_enabled = False

    def __enter__(self):
        self.was_enabled = self.notifier.isEnabled()
        self.notifier.setEnabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled:
            self.notifier.setEnabled(True)


class ConnectionReader(QtCore.QThread):
    """Qt thread for reading messages from worker connection."""

    message_received = Signal(object)
    connection_closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connection: multiprocessing.connection.Connection | None = None

    def set_connection(self, conn):
        self.connection = conn

    def run(self):
        if not self.connection:
            return

        try:
            while True:
                msg = self.connection.recv()
                self.message_received.emit(msg)
        except (EOFError, OSError):
            self.connection_closed.emit()
        except pickle.PickleError as e:
            logger.error(f"Pickle error: {e}")
        finally:
            try:
                self.connection.close()
            except Exception:
                pass


class QtListener(QtCore.QObject):
    """Qt-friendly wrapper for multiprocessing.connection.Listener."""

    family = "AF_INET"
    connection_accepted = Signal(object)
    connection_error = Signal(str)

    def __init__(self, address=None, backlog=1, authkey=None, parent=None):
        super().__init__(parent)

        self._listener = multiprocessing.connection.Listener(
            address, self.family, backlog, authkey
        )

        self._socket = self._listener._listener._socket  # type: ignore
        self._notifier = QtCore.QSocketNotifier(
            self._socket.fileno(), QtCore.QSocketNotifier.Read, self
        )
        self._notifier.activated.connect(self._on_connection_ready)
        self._notifier.setEnabled(True)

    def _on_connection_ready(self):
        with TemporarilyDisableNotifier(self._notifier):
            try:
                ready, _, _ = select.select([self._socket], [], [], 0)
                if not ready:
                    return

                conn = self.accept()
                self.connection_accepted.emit(conn)
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                self.connection_error.emit(str(e))

    def accept(self):
        return self._listener.accept()

    def close(self):
        self._notifier.setEnabled(False)
        self._notifier.deleteLater()
        self._listener.close()

    @property
    def address(self):
        return self._listener.address


class WorkerLauncher(QtCore.QProcess):
    """
    Manages external worker processes using QProcess.
    Provides bidirectional communication via multiprocessing connections.
    """

    # Signals
    processing_results = Signal(dict)
    error_occurred_msg = Signal(str)
    worker_message = Signal(object)
    worker_connected = Signal()
    worker_disconnected = Signal()

    def __init__(self, message_emitter: MessageEmitter | None = None, parent=None):
        super(WorkerLauncher, self).__init__(parent)

        self.message_emitter = message_emitter

        self.readyReadStandardOutput.connect(self._on_stdout)
        self.readyReadStandardError.connect(self._on_stderr)
        self.errorOccurred.connect(self._on_error)
        self.stateChanged.connect(self._on_state_changed)

        self.python_interpreter = MultiprocessingHelper.get_python_interpreter()

        self.listener = None
        self.connection = None
        self.authkey = None

        self.reader_thread = ConnectionReader(self)
        self.reader_thread.message_received.connect(self._on_worker_message)
        self.reader_thread.connection_closed.connect(self._on_connection_closed)

        self.connection_attempts = 0
        self.max_connection_attempts = 10

        self._streams: dict[str, dict] = {}

    def is_not_running(self):
        return self.state() == QtCore.QProcess.NotRunning

    def _on_worker_message(self, message):
        """Process messages from the worker, including chunked streams."""
        # Handle chunked messages
        msg_id = message.get("message_id")
        if msg_id:
            idx = message["chunk_index"]
            total = message["total_chunks"]
            msg_type = message.get("type")

            stream = self._streams.setdefault(
                msg_id, {"type": msg_type, "chunks": {}, "total": total}
            )
            stream["chunks"][idx] = message["data"]
            logger.info(
                "Received chunk %d/%d for %r (id=%s)",
                idx + 1,
                total,
                msg_type,
                msg_id,
            )

            if len(stream["chunks"]) == total:
                full = []
                for i in range(total):
                    full.extend(stream["chunks"][i])
                del self._streams[msg_id]
                logger.info(
                    "%r streaming complete (id=%s, %d items)",
                    msg_type,
                    msg_id,
                    len(full),
                )
                self.processing_results.emit(
                    {
                        "type": "result",
                        "results": full,
                        "status": "success",
                    }
                )
            return

        # Non-chunked messages
        logger.debug("← Received message from worker: %r", message)
        self.worker_message.emit(message)

        # Delegate to MessageEmitter if available
        if self.message_emitter and isinstance(message, dict):
            msg_type = message.get("type")
            if msg_type == "error":
                self.message_emitter.emit_worker_error(
                    message.get("error", "Unknown error")
                )
            elif msg_type == "result":
                self.message_emitter.emit_worker_results(
                    {
                        "results": message.get("data"),
                        "status": message.get("status", "success"),
                    }
                )
            elif msg_type == "status":
                self.message_emitter.emit_worker_message(message)

    def _on_connection_closed(self):
        logger.info("Worker IPC connection closed.")
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
        self.worker_disconnected.emit()

        if self.message_emitter:
            self.message_emitter.emit_worker_disconnected()

    def launch_worker(self, script_path: str, worker_args: dict):
        """
        Starts the worker script with specified arguments.

        Args:
            script_path: Path to the worker script
            worker_args: Dictionary of arguments to pass to the worker
        """
        self._cleanup_resources()

        # Create listener
        self.authkey = os.urandom(32)
        logger.info(f"Generated Authkey: {self.authkey.hex()}")
        self.listener = QtListener(
            ("localhost", 0),
            authkey=self.authkey,
            parent=self,
        )
        address = self.listener.address
        logger.info(f"Created listener on {address}")
        self.listener.connection_accepted.connect(self._on_connection_accepted)
        self.listener.connection_error.connect(self._on_connection_error)

        # Prepare QProcess
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHON_PATH", str(self.python_interpreter.parent))
        env.insert("PYTHON_BIN", str(self.python_interpreter.name))
        self.setProcessEnvironment(env)

        # Build command line arguments
        args = ["-u", script_path]

        # Add connection info
        args.extend(
            [
                "--address",
                f"{address[0]}:{address[1]}",
                "--authkey",
                self.authkey.hex(),
            ]
        )

        # Add worker-specific arguments
        for key, value in worker_args.items():
            args.extend([f"--{key}", str(value)])

        logger.info(f"Starting worker process: {self.python_interpreter} {args}")
        self.start(str(self.python_interpreter), args)

        if not self.waitForStarted(5000):
            logger.error(f"Worker process failed to start: {self.errorString()}")
            self._cleanup_resources()
            return False

        self.connection_attempts = 0
        logger.info("Worker process started. Beginning connection attempts...")
        return True

    def _on_connection_accepted(self, conn):
        logger.info("Connection from worker accepted")
        self.reader_thread.set_connection(conn)
        if not self.reader_thread.isRunning():
            self.reader_thread.start()
        self.connection = conn
        self.worker_connected.emit()

        if self.message_emitter:
            self.message_emitter.emit_worker_connected()

    def _on_connection_error(self, error_msg):
        logger.error(f"Connection error: {error_msg}")
        self.connection_attempts += 1

        if self.connection_attempts >= self.max_connection_attempts:
            self.error_occurred_msg.emit(
                f"Failed to connect to worker after {self.max_connection_attempts} attempts"
            )
            self.stop_worker()

    def _cleanup_resources(self):
        if self.reader_thread.isRunning():
            self.reader_thread.terminate()
            self.reader_thread.wait()

        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None

        if self.listener:
            try:
                self.listener.close()
            except:
                pass
            self.listener = None

    def stop_worker(self):
        """Attempts to terminate the worker process gracefully."""
        if self.is_not_running() and not self.connection:
            logger.debug("Worker process was already stopped and connection closed.")
            return

        logger.info("Attempting to stop worker process...")

        if self.connection:
            try:
                logger.info("Sending exit command...")
                self.send_command({"command": "exit"})
                if self.waitForFinished(1000):
                    logger.info("Worker exited gracefully.")
                    self._cleanup_resources()
                    return
            except Exception as e:
                logger.error(f"Error sending exit command: {e}")

        if not self.is_not_running():
            logger.warning("Worker did not exit gracefully, terminating...")
            self.terminate()
            if not self.waitForFinished(2000):
                logger.warning("Worker did not terminate, killing...")
                self.kill()
                self.waitForFinished(1000)

        self._cleanup_resources()
        logger.info("Worker process shutdown complete.")

    def send_command(self, command):
        """Sends a command object via the connection."""
        if not self.connection:
            logger.warning(
                f"Cannot send command '{command}', IPC connection not established."
            )
            return False

        logger.debug(f"→ Sending command: {command}")
        try:
            self.connection.send(command)
            logger.debug(f"→ Successfully sent command: {command}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command '{command}': {e}")
            self._on_connection_closed()
            return False

    def _on_stdout(self):
        out_bytes = self.readAllStandardOutput()
        if not out_bytes:
            return
        out = out_bytes.data().decode("utf-8", errors="replace")
        if out.strip():
            print(out.strip(), flush=True)

    def _on_stderr(self):
        data = self.readAllStandardError().data()
        if data:
            data = data.decode("utf-8", errors="replace").strip()
            print(data.strip(), file=sys.stderr, flush=True)

    def _on_error(self, error: QtCore.QProcess.ProcessError):
        error_map = {
            QtCore.QProcess.FailedToStart: "FailedToStart",
            QtCore.QProcess.Crashed: "Crashed",
            QtCore.QProcess.Timedout: "Timedout",
            QtCore.QProcess.ReadError: "ReadError",
            QtCore.QProcess.WriteError: "WriteError",
            QtCore.QProcess.UnknownError: "UnknownError",
        }
        error_str = error_map.get(error, f"UnknownError({error})")
        msg = f"Worker process error: {error_str} - {self.errorString()}"
        print(msg.strip(), file=sys.stderr, flush=True)
        self.error_occurred_msg.emit(msg)

    def _on_state_changed(self, state: QtCore.QProcess.ProcessState):
        state_map = {
            QtCore.QProcess.NotRunning: "NotRunning",
            QtCore.QProcess.Starting: "Starting",
            QtCore.QProcess.Running: "Running",
        }
        state_str = state_map.get(state, f"UnknownState({state})")
        logger.info(f"Worker process state changed: {state_str}")

        if self.is_not_running():
            self._cleanup_resources()
            exit_code = self.exitCode()
            exit_status = self.exitStatus()
            exit_status_str = (
                "NormalExit"
                if exit_status == QtCore.QProcess.NormalExit
                else "CrashExit"
            )
            msg = f"Worker process exited with code {exit_code} ({exit_status_str})"
            logger.info(msg)
            if exit_status == QtCore.QProcess.CrashExit:
                self.error_occurred_msg.emit(msg)
