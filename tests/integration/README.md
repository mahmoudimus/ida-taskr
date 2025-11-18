# IDA Taskr Integration Tests

This directory contains integration tests for IDA Taskr's Qt Core functionality.

## Overview

The integration tests verify that IDA Taskr's Qt-based process management and threading work correctly with PySide6 in headless mode.

## Running Tests Locally

### Prerequisites

Install system dependencies (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0
```

### Running Tests

```bash
# Install Python dependencies
pip install -e .[ci]

# Run all Qt Core integration tests
QT_QPA_PLATFORM=offscreen python -m pytest tests/integration/test_integration_qt_core.py -v

# Run a specific test class
QT_QPA_PLATFORM=offscreen python -m pytest tests/integration/test_integration_qt_core.py::TestQtCoreFramework -v

# Run a specific test method
QT_QPA_PLATFORM=offscreen python -m pytest tests/integration/test_integration_qt_core.py::TestQtCoreFramework::test_qt_framework_import -v
```

## CI/CD

Integration tests run automatically in GitHub Actions on:
- Pull requests to main
- Pushes to main

The workflow:
1. Installs system dependencies (OpenGL, X11 libraries)
2. Installs Python dependencies including PySide6
3. Runs integration tests in headless mode with QT_QPA_PLATFORM=offscreen
4. Uploads coverage reports

## Test Structure

### Test Files

- `test_integration_qt_core.py` - Qt Core functionality tests
  - Tests WorkerLauncher (QProcess-based process spawning)
  - Tests MessageEmitter (Qt signal/slot communication)
  - Tests TaskRunner Qt integration
  - Tests cross-thread signaling
  - Tests QProcessEnvironment
  - All tests run in headless mode with PySide6

### Configuration

- `conftest.py` - Pytest fixtures and configuration

### Fixtures

- `qtbot` - pytest-qt fixture for managing QApplication and testing Qt components (provided by pytest-qt plugin)

## Adding New Tests

To add new Qt Core integration tests:

1. Add test methods to existing test classes in `test_integration_qt_core.py`
2. Import PySide6 components at module level
3. Use the `qtbot` fixture for QApplication management
4. Run tests in headless mode with `QT_QPA_PLATFORM=offscreen`

Example:

```python
from PySide6.QtCore import QObject, Signal

class TestMyNewFeature:
    def test_custom_signal_handling(self, qtbot):
        """Test custom Qt signal handling."""
        class CustomEmitter(QObject):
            my_signal = Signal(int)

        emitter = CustomEmitter()
        received = []

        def handler(value):
            received.append(value)

        emitter.my_signal.connect(handler)
        emitter.my_signal.emit(42)

        qtbot.wait(10)
        assert received == [42]
```

## Troubleshooting

### ImportError: libGL.so.1 cannot open shared object file

Install the required system dependencies:

```bash
sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dev libegl1
```

### Qt Application Errors

Ensure `QT_QPA_PLATFORM=offscreen` is set:

```bash
export QT_QPA_PLATFORM=offscreen
python -m pytest tests/integration/test_integration_qt_core.py -v
```

### Tests Hang

If tests hang, check for:
1. Infinite loops in Qt event processing
2. Signals not being emitted
3. QProcess not terminating - use `waitForFinished()` with timeout

## Performance Considerations

- Integration tests are slower than unit tests (5-30s per test file)
- QProcess spawning adds overhead
- Use timeouts on `waitForStarted()` and `waitForFinished()`
- Consider using `@pytest.mark.timeout(30)` for tests that might hang

## Coverage

Integration test coverage is tracked separately from unit test coverage. Coverage reports are uploaded as artifacts in CI/CD runs and can be viewed after workflow completion.
