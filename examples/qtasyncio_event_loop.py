"""
Example demonstrating QtAsyncio event loop integration with async/await.

This example shows how to use the QtAsyncio module to integrate Python's
asyncio with Qt's event loop, enabling natural async/await syntax in Qt
applications.
"""

import asyncio
import sys
import time

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
    from PySide6.QtCore import Qt, QTimer
except ImportError:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
    from PyQt5.QtCore import Qt, QTimer

from ida_taskr import (
    QT_ASYNCIO_AVAILABLE,
    get_logger,
)

if not QT_ASYNCIO_AVAILABLE:
    print("QtAsyncio module not available. Please ensure Qt (PyQt5 or PySide6) is installed.")
    sys.exit(1)

from ida_taskr import (
    set_event_loop_policy,
    qtasyncio_run,
    ThreadExecutor,
)

logger = get_logger(__name__)

# Set the Qt-compatible event loop policy
set_event_loop_policy()


# Example async functions
async def async_countdown(name, count):
    """Async countdown that yields control to the event loop."""
    logger.info(f"{name}: Starting countdown from {count}")
    for i in range(count, 0, -1):
        logger.info(f"{name}: {i}")
        await asyncio.sleep(1)
    logger.info(f"{name}: Done!")
    return f"{name} completed"


async def async_fetch_data(delay=2):
    """Simulate async data fetching."""
    logger.info("Fetching data...")
    await asyncio.sleep(delay)
    logger.info("Data fetched!")
    return {"status": "success", "data": [1, 2, 3, 4, 5]}


async def async_concurrent_tasks():
    """Run multiple async tasks concurrently."""
    logger.info("Starting concurrent tasks...")

    # Run multiple tasks concurrently
    results = await asyncio.gather(
        async_countdown("Task-A", 3),
        async_countdown("Task-B", 5),
        async_fetch_data(2),
    )

    logger.info(f"All tasks complete: {results}")
    return results


class AsyncioExampleWindow(QMainWindow):
    """Demo window showing asyncio integration with Qt."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QtAsyncio Event Loop Examples")
        self.setGeometry(100, 100, 500, 300)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status label
        self.status_label = QLabel("Ready - Click a button to run async tasks")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Example 1: Simple async countdown
        btn1 = QPushButton("Run Simple Async Task")
        btn1.clicked.connect(lambda: asyncio.create_task(self.run_simple_async()))
        layout.addWidget(btn1)

        # Example 2: Concurrent async tasks
        btn2 = QPushButton("Run Concurrent Async Tasks")
        btn2.clicked.connect(lambda: asyncio.create_task(self.run_concurrent_async()))
        layout.addWidget(btn2)

        # Example 3: ThreadExecutor integration
        btn3 = QPushButton("Run ThreadExecutor Task")
        btn3.clicked.connect(self.run_thread_executor)
        layout.addWidget(btn3)

        # Example 4: Mixing Qt timers with async
        btn4 = QPushButton("Mix Qt Timer with Async")
        btn4.clicked.connect(self.run_mixed_qt_async)
        layout.addWidget(btn4)

        # Create a ThreadExecutor instance
        self.executor = ThreadExecutor(self)

    async def run_simple_async(self):
        """Example 1: Run a simple async task."""
        self.status_label.setText("Running simple async countdown...")
        result = await async_countdown("Simple", 5)
        self.status_label.setText(f"Complete: {result}")

    async def run_concurrent_async(self):
        """Example 2: Run multiple concurrent async tasks."""
        self.status_label.setText("Running concurrent async tasks...")
        results = await async_concurrent_tasks()
        self.status_label.setText(f"All tasks complete! Results: {len(results)} tasks")

    def run_thread_executor(self):
        """Example 3: Use ThreadExecutor for CPU-bound tasks."""
        self.status_label.setText("Running CPU-bound task in ThreadExecutor...")

        def cpu_intensive_task(n):
            """Simulate CPU-intensive work."""
            logger.info(f"Computing sum of squares up to {n}...")
            result = sum(i * i for i in range(n))
            time.sleep(2)  # Simulate more work
            return result

        # Submit task to executor
        future = self.executor.submit(cpu_intensive_task, 1000000)

        # Use asyncio to wait for the future
        asyncio.create_task(self.wait_for_future(future))

    async def wait_for_future(self, future):
        """Wait for a concurrent.futures.Future in async context."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, future.result)
        self.status_label.setText(f"ThreadExecutor result: {result:,}")
        logger.info(f"Computation complete: {result:,}")

    def run_mixed_qt_async(self):
        """Example 4: Mix Qt signals/timers with async/await."""
        self.status_label.setText("Mixing Qt timer with async...")

        # Create a Qt timer
        timer = QTimer(self)
        countdown = [5]  # Use list for mutability in closure

        async def async_with_timer():
            """Async function that works with Qt timer."""
            for i in range(5):
                await asyncio.sleep(1)
                self.status_label.setText(f"Async tick {i + 1}/5")

            timer.stop()
            self.status_label.setText("Mixed Qt/Async complete!")

        def on_timer():
            """Qt timer callback."""
            countdown[0] -= 1
            logger.info(f"Qt Timer tick: {countdown[0]}")
            if countdown[0] == 0:
                timer.stop()

        # Start Qt timer
        timer.timeout.connect(on_timer)
        timer.start(1000)

        # Start async task
        asyncio.create_task(async_with_timer())


async def async_main():
    """Async main function that runs the Qt application."""
    app = QApplication(sys.argv)
    window = AsyncioExampleWindow()
    window.show()

    # Run the Qt event loop asynchronously
    # This allows mixing Qt and asyncio seamlessly
    await asyncio.Event().wait()  # Wait forever (app.quit() will exit)


def main():
    """Run the asyncio/Qt integrated application."""
    # Note: When using QtAsyncio, you can use asyncio.run() directly
    # The event loop policy we set ensures Qt integration
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")


if __name__ == "__main__":
    main()
