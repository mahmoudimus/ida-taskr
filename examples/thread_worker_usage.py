"""
Example demonstrating the thread_worker decorator and Qt-based worker utilities.

This example shows how to use the qtasyncio module's worker utilities for
running tasks in background threads with Qt signal integration.
"""

import sys
import time

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
    from PySide6.QtCore import Qt
except ImportError:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
    from PyQt5.QtCore import Qt

from ida_taskr import (
    QT_ASYNCIO_AVAILABLE,
    get_logger,
)

if not QT_ASYNCIO_AVAILABLE:
    print("QtAsyncio module not available. Please ensure Qt (PyQt5 or PySide6) is installed.")
    sys.exit(1)

from ida_taskr import thread_worker, create_worker, FunctionWorker

logger = get_logger(__name__)


# Example 1: Simple function worker using the decorator
@thread_worker
def compute_fibonacci(n):
    """Compute fibonacci number (slow recursive version for demo)."""
    if n <= 1:
        return n
    return compute_fibonacci.__wrapped__(n - 1) + compute_fibonacci.__wrapped__(n - 2)


# Example 2: Long-running task with progress
def long_running_task(duration=5):
    """Simulate a long-running task."""
    logger.info(f"Starting task that will run for {duration} seconds...")
    for i in range(duration):
        time.sleep(1)
        logger.info(f"Progress: {i+1}/{duration}")
    logger.info("Task complete!")
    return f"Completed after {duration} seconds"


# Example 3: Generator worker (yields intermediate results)
def generate_numbers(start, end):
    """Generate numbers and yield each one."""
    for i in range(start, end + 1):
        time.sleep(0.1)  # Simulate work
        yield i


class WorkerExampleWindow(QMainWindow):
    """Demo window showing different worker patterns."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QtAsyncio Worker Examples")
        self.setGeometry(100, 100, 400, 300)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Example 1: Decorated function worker
        btn1 = QPushButton("Run Decorated Worker (Fibonacci)")
        btn1.clicked.connect(self.run_decorated_worker)
        layout.addWidget(btn1)

        # Example 2: Manual worker creation
        btn2 = QPushButton("Run Manual Worker (Long Task)")
        btn2.clicked.connect(self.run_manual_worker)
        layout.addWidget(btn2)

        # Example 3: Generator worker
        btn3 = QPushButton("Run Generator Worker")
        btn3.clicked.connect(self.run_generator_worker)
        layout.addWidget(btn3)

    def run_decorated_worker(self):
        """Example 1: Using the @thread_worker decorator."""
        self.status_label.setText("Computing Fibonacci(35)...")

        # Create worker using decorated function
        worker = compute_fibonacci(35)

        # Connect signals
        worker.returned.connect(self.on_fib_result)
        worker.errored.connect(self.on_error)
        worker.finished.connect(lambda: logger.info("Fibonacci worker finished"))

        # Start the worker
        worker.start()

    def run_manual_worker(self):
        """Example 2: Manually creating a FunctionWorker."""
        self.status_label.setText("Running long task...")

        # Create worker manually
        worker = create_worker(long_running_task, duration=3)

        # Connect signals
        worker.returned.connect(self.on_task_result)
        worker.errored.connect(self.on_error)
        worker.finished.connect(lambda: logger.info("Long task worker finished"))

        # Start the worker
        worker.start()

    def run_generator_worker(self):
        """Example 3: Using a generator worker that yields intermediate results."""
        self.status_label.setText("Generating numbers...")

        # Create generator worker
        worker = create_worker(generate_numbers, 1, 10)

        # Connect to yielded signal for intermediate results
        worker.yielded.connect(self.on_number_yielded)
        worker.returned.connect(self.on_generator_complete)
        worker.errored.connect(self.on_error)

        # Start the worker
        worker.start()

    def on_fib_result(self, result):
        """Handle fibonacci computation result."""
        self.status_label.setText(f"Fibonacci(35) = {result}")
        logger.info(f"Fibonacci result: {result}")

    def on_task_result(self, result):
        """Handle long task result."""
        self.status_label.setText(result)
        logger.info(f"Task result: {result}")

    def on_number_yielded(self, number):
        """Handle each number yielded by the generator."""
        self.status_label.setText(f"Generated: {number}")
        logger.info(f"Yielded: {number}")

    def on_generator_complete(self, result):
        """Handle generator completion."""
        self.status_label.setText(f"Generator complete! Final value: {result}")
        logger.info(f"Generator complete with result: {result}")

    def on_error(self, error):
        """Handle worker errors."""
        self.status_label.setText(f"Error: {error}")
        logger.error(f"Worker error: {error}")


def main():
    """Run the worker examples."""
    app = QApplication(sys.argv)
    window = WorkerExampleWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
