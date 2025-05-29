"""Protocol definitions and base classes for the worker manager."""

from abc import ABC, abstractmethod

from .utils import EventEmitter


class WorkerProtocol(ABC):
    """Protocol for worker implementations."""

    @abstractmethod
    def setup(self, **kwargs):
        """Set up the worker."""
        pass

    @abstractmethod
    def process(self, connection, **kwargs):
        """Process tasks."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class MessageEmitter(EventEmitter):
    """Event emitter for handling messages from workers in IDA.

    Events emitted:
    - 'worker_connected': When worker establishes connection
    - 'worker_message': When a message is received from worker (payload: message dict)
    - 'worker_results': When results are received from worker (payload: results dict)
    - 'worker_error': When an error occurs (payload: error string)
    - 'worker_disconnected': When worker connection is closed
    """

    def emit_worker_connected(self):
        """Emit worker connected event."""
        self.emit("worker_connected")

    def emit_worker_message(self, message: dict):
        """Emit worker message event."""
        self.emit("worker_message", message)

    def emit_worker_results(self, results: dict):
        """Emit worker results event."""
        self.emit("worker_results", results)

    def emit_worker_error(self, error: str):
        """Emit worker error event."""
        self.emit("worker_error", error)

    def emit_worker_disconnected(self):
        """Emit worker disconnected event."""
        self.emit("worker_disconnected")
