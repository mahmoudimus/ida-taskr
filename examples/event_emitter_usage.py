"""
Example showing how to use the MessageEmitter pattern for handling worker messages.

This demonstrates the composition-based approach which is flexible and easy to test.
"""

import logging

from ida_taskr import TaskRunner, get_logger

logger = get_logger(__name__)


def on_results(results):
    logger.info("✅ Received results: %s", results)
    if results.get("status") == "success":
        data = results.get("results", [])
        logger.info("🎯 Processing %d result items", len(data))


def on_progress(progress, status):
    logger.info("📊 Progress: %.1f%% - %s", progress * 100, status)


def main():
    runner = TaskRunner(
        worker_script="path/to/your/worker.py",
        worker_args={"data_size": 1024, "start_ea": "0x1000", "is64": "1"},
        log_level=logging.DEBUG,
    )
    runner.on_results(on_results)
    runner.on_progress(on_progress)
    runner.start()


if __name__ == "__main__":
    main()
