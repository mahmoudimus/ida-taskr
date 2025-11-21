# Running Tests in IDA Pro

Some tests in ida-taskr require IDA Pro's Qt application to be running. These tests are marked with `@pytest.mark.skipif(not is_ida())` and will automatically skip when run outside IDA.

## Tests That Require IDA Pro

### WorkerLauncher Integration Test
**File:** `tests/unit/test_event_emitter.py`
**Test:** `TestMessageEmitter::test_worker_launcher_integration`
**Reason:** `WorkerLauncher` inherits from `QProcess` and requires a full Qt event loop with socket notifiers.

### Worker Execution Test
**File:** `tests/unit/test_qtasyncio.py`
**Test:** `TestQtApplicationIntegration::test_full_worker_execution`
**Reason:** Worker utilities need a running Qt application to create QThread-based workers.

## How to Run Tests in IDA Pro

### Method 1: IDA Python Console

```python
# Inside IDA Pro's Python console
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/unit/test_event_emitter.py::TestMessageEmitter::test_worker_launcher_integration",
    "tests/unit/test_qtasyncio.py::TestQtApplicationIntegration::test_full_worker_execution",
    "-v"
])
```

### Method 2: IDAPython Script

1. Save as `run_tests.py` in your IDA scripts directory
2. File → Script file... → Select `run_tests.py`

```python
import subprocess
import sys

result = subprocess.run([
    sys.executable, "-m", "pytest",
    "tests/",  # Run all tests
    "-v",
    "--tb=short"
])

print(f"Tests {'passed' if result.returncode == 0 else 'failed'}")
```

### Method 3: Headless IDA (Command Line)

```bash
# Run IDA in headless mode with test script
idat64 -A -S"tests/run_ida_tests.py" /path/to/binary

# Or run pytest directly with IDA's Python
/path/to/ida/python -m pytest tests/ -v -k "not skipif"
```

### Method 4: Using the Helper Script

```bash
# From your terminal (will skip IDA-only tests)
./run_tests.sh

# Results:
# - 139 tests pass
# - 2 tests skip (IDA-only)
```

Then run IDA-only tests inside IDA:
```python
# In IDA Pro:
exec(open('tests/run_ida_tests.py').read())
```

## Test Status

| Test | Outside IDA | Inside IDA |
|------|-------------|------------|
| Regular unit tests | ✅ Pass (139) | ✅ Pass |
| `test_worker_launcher_integration` | ⏭️ Skip | ✅ Pass |
| `test_full_worker_execution` | ⏭️ Skip | ✅ Pass |

## CI/CD

The GitHub Actions workflow runs tests in two jobs:

1. **unit-tests** - Runs all regular tests (139 pass, 2 skip)
2. **qt-integration-tests** - Runs Qt integration tests with `QT_QPA_PLATFORM=offscreen`

IDA-specific tests are intentionally skipped in CI since they require actual IDA Pro installation.

## Troubleshooting

### "WorkerLauncher requires IDA Pro's Qt application"
This means you're running the test outside IDA. The test will automatically skip. Run it inside IDA Pro to execute.

### Segmentation Fault
If you see a segfault when running `WorkerLauncher` tests, ensure you're running inside IDA Pro where the Qt application is properly initialized.

### "Qt not available"
Make sure PySide6 or PyQt5 is installed in IDA's Python environment:
```bash
/path/to/ida/python -m pip install PySide6
```
