"""Generic IDA Pro plugin for taskr framework."""

import idaapi

from . import DataProcessorCore, get_logger

logger = get_logger(__name__)


class DataProcessorPlugin(idaapi.plugin_t):
    """Generic IDA Pro plugin for data processing tasks."""

    flags = idaapi.PLUGIN_PROC
    comment = "Generic Data Processor via Shared Memory and Multiprocessing"
    help = "Press Alt-Shift-P to start processing"
    wanted_name = "TaskrProcessor"
    wanted_hotkey = "Alt-Shift-P"
    _core: DataProcessorCore | None = None

    def init(self):
        # This will be overridden by specific implementations
        logger.info("Generic TaskrProcessor plugin initialized")
        return idaapi.PLUGIN_KEEP

    def term(self):
        logger.info("Terminating plugin.")
        if self._core:
            self._core.terminate()
        logger.info("Plugin terminated.")

    def run(self, arg):
        # This will be overridden by specific implementations
        logger.warning("Generic plugin run() called - should be overridden")


def PLUGIN_ENTRY():
    return DataProcessorPlugin()


# XXX: This is a temporary hack to allow the plugin to be loaded
"""
from .ida_plugin import PLUGIN_ENTRY, DataProcessorPlugin

__all__ = ["DataProcessorPlugin", "PLUGIN_ENTRY"]
from ida_worker_manager import ConnectionContext, WorkerBase


class MyWorker(WorkerBase):
    def process(self, connection: ConnectionContext, **kwargs):
        # Your processing logic here
        connection.send_message("status", "processing")
        # Do work...
        connection.send_message("result", {"data": "results"})


# In your IDA plugin:
from ida_worker_manager import MessageHandler, WorkerLauncher


class MyHandler(MessageHandler):
    def on_worker_results(self, results):
        # Handle results
        pass


launcher = WorkerLauncher(MyHandler())
launcher.launch_worker("my_worker.py", {"arg1": "value1"})
"""
