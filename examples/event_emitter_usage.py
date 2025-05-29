"""
Example showing how to use the MessageEmitter pattern for handling worker messages.

This demonstrates the composition-based approach which is flexible and easy to test.
"""

from ida_taskr import MessageEmitter, WorkerLauncher, get_logger

logger = get_logger(__name__)


def create_message_emitter():
    """Create a message emitter with custom event handlers.

    Returns:
        MessageEmitter configured with event handlers
    """
    # Create the emitter
    emitter = MessageEmitter()

    # Set up event handlers using the decorator syntax
    @emitter.on("worker_connected")
    def on_connected():
        logger.info("🔗 Worker connected successfully!")

    @emitter.on("worker_message")
    def on_message(message: dict):
        logger.info("📨 Received message: %s", message)

        # You can handle different message types
        if message.get("type") == "progress":
            progress = message.get("progress", 0)
            logger.info("📊 Progress: %.1f%%", progress * 100)

    @emitter.on("worker_results")
    def on_results(results: dict):
        logger.info("✅ Received results: %s", results)
        # Process your results here
        if results.get("status") == "success":
            data = results.get("results", [])
            logger.info("🎯 Processing %d result items", len(data))

    @emitter.on("worker_error")
    def on_error(error: str):
        logger.error("❌ Worker error: %s", error)

    @emitter.on("worker_disconnected")
    def on_disconnected():
        logger.info("🔌 Worker disconnected")

    return emitter


def create_message_emitter_alternative():
    """Alternative way to create a message emitter by registering handlers directly.

    Returns:
        MessageEmitter configured with event handlers
    """
    emitter = MessageEmitter()

    # You can also register handlers without decorators
    def handle_connection():
        logger.info("Worker is now connected")

    def handle_message(message: dict):
        logger.info("Got message: %s", message)

    def handle_results(results: dict):
        logger.info("Got results: %s", results)

    def handle_error(error: str):
        logger.error("Error occurred: %s", error)

    def handle_disconnection():
        logger.info("Worker disconnected")

    # Register the handlers
    emitter.on("worker_connected", handle_connection)
    emitter.on("worker_message", handle_message)
    emitter.on("worker_results", handle_results)
    emitter.on("worker_error", handle_error)
    emitter.on("worker_disconnected", handle_disconnection)

    return emitter


def create_multiple_subscribers_example():
    """Example showing multiple subscribers to the same event."""
    emitter = MessageEmitter()

    # Multiple handlers for the same event
    @emitter.on("worker_results")
    def log_results(results: dict):
        logger.info("📝 Logging results: %d items", len(results.get("results", [])))

    @emitter.on("worker_results")
    def process_results(results: dict):
        logger.info("⚙️  Processing results...")
        # Actually process the results
        for item in results.get("results", []):
            # process_item(item)
            pass

    @emitter.on("worker_results")
    def notify_ui(results: dict):
        logger.info("🖥️  Updating UI with results")
        # Update UI with results
        # ui.update_progress(100)

    return emitter


def main():
    """Example usage of the MessageEmitter pattern."""

    # Create a message emitter with event handlers
    message_emitter = create_message_emitter()

    # Create worker launcher with the message emitter
    launcher = WorkerLauncher(message_emitter)

    # Launch your worker
    worker_args = {"data_size": 1024, "start_ea": "0x1000", "is64": "1"}

    if launcher.launch_worker("path/to/your/worker.py", worker_args):
        logger.info("Worker launched successfully")
    else:
        logger.error("Failed to launch worker")


if __name__ == "__main__":
    main()
