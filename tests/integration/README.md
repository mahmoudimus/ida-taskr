# IDA Taskr Integration Tests

This directory contains integration tests for IDA Taskr that run against real IDA Pro instances.

## Overview

The integration tests verify that IDA Taskr works correctly with actual IDA Pro installations, including:
- **IDA Pro 9.1** with PyQt5 ✅
- **IDA Pro 9.2** with PySide6 ✅

## Running Tests Locally

### Using Docker Compose

The easiest way to run integration tests is using Docker Compose:

```bash
# Run tests for IDA 9.1 (PyQt5)
docker compose run --rm idapro-tests

# Run tests for IDA 9.2 (PySide6)
docker compose run --rm idapro-tests-9.2

# Run both in sequence
docker compose run --rm idapro-tests && docker compose run --rm idapro-tests-9.2
```

### Running Specific Tests

To run specific test files or test cases:

```bash
# Run Qt Core tests only (no IDA required, faster)
docker compose run --rm --entrypoint bash idapro-tests -c "pip install -e .[ci] && python -m pytest tests/integration/test_integration_qt_core.py -v"

# Run IDA-specific tests only
docker compose run --rm --entrypoint bash idapro-tests -c "pip install -e .[ci] && python -m pytest tests/integration/test_integration_ida.py -v"

# Run a specific test class (IDA 9.2)
docker compose run --rm --entrypoint bash idapro-tests-9.2 -c "pip install -e .[ci] && python -m pytest tests/integration/test_integration_ida.py::TestTaskRunnerIntegration -v"

# Run a specific test method
docker compose run --rm --entrypoint bash idapro-tests -c "pip install -e .[ci] && python -m pytest tests/integration/test_integration_qt_core.py::TestQtCoreFramework::test_qt_framework_import -v"
```

## CI/CD

Integration tests run automatically in GitHub Actions on:
- Pull requests to main
- Pushes to main

The workflow:
1. Pulls the appropriate IDA Pro Docker image
2. Runs integration tests in isolated containers
3. Uploads test artifacts and coverage reports
4. Caches Docker images for faster subsequent runs

## Test Structure

### Test Files

Integration tests are organized into two categories:

#### Qt Core Tests (No IDA Required)
- `test_integration_qt_core.py` - Tests for Qt Core functionality (QProcess, QThread, signals/slots)
  - Tests WorkerLauncher, MessageEmitter, and Qt-based process management
  - Runs in headless mode with both PyQt5 and PySide6
  - Does not require IDA Pro - only Qt framework

#### IDA Pro Tests (Require IDA)
- `test_integration_ida.py` - Tests for TaskRunner functionality in IDA environment
- `test_integration_ida_analysis.py` - Tests for IDA analysis capabilities
  - Require actual IDA Pro installation
  - Test IDA API access and binary analysis

### Configuration

- `conftest.py` - Pytest fixtures and configuration

### Fixtures

- `ida_available` - Checks if IDA Pro is available
- `qt_framework` - Determines which Qt framework is available (PyQt5 or PySide6) in IDA GUI mode
- `qt_framework_headless` - Determines Qt framework without requiring IDA GUI mode
- `test_binary` - Provides a test binary for analysis
- `temp_idb` - Creates a temporary IDA database
- `ida_database` - Opens and analyzes an IDA database
- `output_dir` - Provides path to output directory for test artifacts
- `qtbot` - pytest-qt fixture for managing QApplication and testing Qt components

## Adding New Tests

To add new integration tests:

1. Create a new test file in `tests/integration/`
2. Use the provided fixtures from `conftest.py`
3. Ensure tests work with both PyQt5 and PySide6
4. Add test binaries to `tests/_resources/bin/` if needed

Example:

```python
import pytest


class TestMyNewFeature:
    def test_feature_with_ida(self, ida_available, ida_database):
        """Test my new feature with IDA Pro."""
        if not ida_database:
            pytest.skip("IDA database not available")

        import idaapi

        # Your test code here
        assert True
```

## Qt Framework Compatibility

The tests are designed to work with both Qt frameworks:

- **IDA 9.1**: Uses PyQt5
  - Signal: `pyqtSignal`
  - Slot: `pyqtSlot`

- **IDA 9.2**: Uses PySide6
  - Signal: `Signal`
  - Slot: `Slot`

When writing tests, use the `qt_framework` fixture to handle differences:

```python
def test_qt_signals(self, qt_framework):
    if qt_framework == "PyQt5":
        from PyQt5.QtCore import QObject, pyqtSignal as Signal
    else:
        from PySide6.QtCore import QObject, Signal

    # Your test code here
```

## Test Resources

Test binaries should be placed in `tests/_resources/bin/`. The integration tests will automatically discover and use these binaries.

Supported binary formats:
- Windows PE (.exe, .dll)
- Linux ELF (.elf, .so)
- Raw binaries (.bin)

## Troubleshooting

### Tests Skip with "IDA database not available"

This usually means:
1. No test binary was found in `tests/_resources/bin/`
2. The binary couldn't be opened by IDA
3. idalib is not available in the IDA installation

Solution: Add a valid test binary to `tests/_resources/bin/`

### Qt Framework Errors

If you see Qt-related errors:
1. Check that the correct Qt framework is installed in the IDA container
2. Verify `QT_QPA_PLATFORM=offscreen` is set in the environment
3. Ensure X11 display is properly configured (for GUI tests)

### Docker Authentication Errors

If you can't pull the IDA Pro Docker images:
1. Ensure you're logged into GitHub Container Registry
2. Verify you have access to the `ghcr.io/mahmoudimus/idapro-linux` images
3. Check that `GITHUB_TOKEN` is properly configured in CI

```bash
# Login locally
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## Performance Considerations

- Integration tests are slower than unit tests (30s - 5min per test file)
- IDA auto-analysis takes time on larger binaries
- Use small test binaries when possible
- Consider using `@pytest.mark.slow` for very long-running tests

## Coverage

Integration test coverage is tracked separately from unit test coverage. Coverage reports are uploaded as artifacts in CI/CD runs and can be viewed after workflow completion.
