"""Worker-side components for the task worker manager."""

import math
import multiprocessing
import multiprocessing.connection
import time
import typing
import uuid

from .helpers import get_logger
from .protocols import WorkerProtocol

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


class WorkerBase(WorkerProtocol):
    """Base class for worker implementations."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._running = False
        self._paused = False

    def setup(self, **kwargs):
        """Default setup implementation - override in subclasses."""
        pass

    def cleanup(self):
        """Default cleanup implementation - override in subclasses."""
        pass

    def handle_command(self, cmd: dict, conn: ConnectionContext) -> bool:
        """
        Handle standard commands. Override for custom commands.
        Returns True to continue processing, False to exit.
        """
        raise NotImplementedError("Subclasses must implement handle_command()")

    def process(self, connection: ConnectionContext, **kwargs):
        """Main processing loop - implement in subclasses."""
        raise NotImplementedError("Subclasses must implement process()")
