"""
TaskRunner - A simplified interface for running worker tasks.

Provides a high-level API that wraps WorkerLauncher and MessageEmitter
for easier task execution with callback-based result handling.
"""

import logging

from .helpers import get_logger
from .launcher import WorkerLauncher
from .protocols import MessageEmitter


class TaskRunner:
    """Simplified task runner with callback-based event handling."""

    def __init__(self, worker_script, worker_args, log_level=None, logger=None):
        """Initialize TaskRunner.

        Args:
            worker_script: Path to the worker script to execute
            worker_args: Dictionary of arguments to pass to the worker
            log_level: Logging level (optional)
            logger: Custom logger instance (optional)
        """
        if logger:
            self.logger = logger
        else:
            # Use INFO as default if no log_level specified
            actual_log_level = log_level if log_level is not None else logging.INFO
            self.logger = get_logger(log_level=actual_log_level)
        self.message_emitter = MessageEmitter()
        self.launcher = WorkerLauncher(self.message_emitter)
        self.worker_script = worker_script
        self.worker_args = worker_args
        self._results_callback = None
        self._progress_callback = None

    def on_results(self, callback):
        """Register a callback for when worker results are received.

        Args:
            callback: Function to call with results dict
        """
        self._results_callback = callback
        self.message_emitter.on("worker_results", self._handle_results)

    def on_progress(self, callback):
        """Register a callback for progress updates.

        Args:
            callback: Function to call with (progress, status) tuple
        """
        self._progress_callback = callback
        self.message_emitter.on("worker_message", self._handle_progress)

    def start(self):
        """Start the worker task."""
        if self.launcher.launch_worker(self.worker_script, self.worker_args):
            self.logger.info("Worker launched successfully")
        else:
            self.logger.error("Failed to launch worker")

    def _handle_results(self, results):
        """Internal handler for worker results."""
        if self._results_callback:
            self._results_callback(results)

    def _handle_progress(self, message):
        """Internal handler for progress messages."""
        if message.get("type") == "progress" and self._progress_callback:
            progress = message.get("progress", 0)
            status = message.get("status", "unknown")
            self._progress_callback(progress, status)
