#!/usr/bin/env python3
"""
Run ida-taskr tests inside IDA Pro.

This script runs pytest tests that require IDA Pro's Qt application.
Some tests are marked with @pytest.mark.skipif(not is_ida()) and will
only run when executed inside IDA Pro.

Usage from IDA Pro:
    1. Via IDA's Python console:
       >>> exec(open('tests/run_ida_tests.py').read())

    2. Via IDAPython script:
       File -> Script file... -> Select this file

    3. Via command line (headless):
       idat64 -A -S"tests/run_ida_tests.py" <binary>

Tests that require IDA Pro's Qt application:
- test_event_emitter.py::TestMessageEmitter::test_worker_launcher_integration
- test_qtasyncio.py::TestQtApplicationIntegration::test_full_worker_execution

These tests will be SKIPPED when run outside IDA Pro.
"""

import sys
import subprocess


def main():
    """Run pytest with tests that require IDA Pro."""
    print("=" * 70)
    print("Running ida-taskr tests inside IDA Pro")
    print("=" * 70)
    print()

    # Tests that require IDA Pro (will be skipped outside IDA)
    ida_tests = [
        "tests/unit/test_event_emitter.py::TestMessageEmitter::test_worker_launcher_integration",
        "tests/unit/test_qtasyncio.py::TestQtApplicationIntegration::test_full_worker_execution",
    ]

    # You can also run all tests - the ones that need IDA will run, others will be skipped if they fail
    all_tests = ["tests/"]

    print("Running IDA-specific tests:")
    for test in ida_tests:
        print(f"  - {test}")
    print()

    # Run pytest
    cmd = [sys.executable, "-m", "pytest"] + ida_tests + ["-v", "--tb=short"]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd="/path/to/ida-taskr")

    print()
    print("=" * 70)
    if result.returncode == 0:
        print("✅ All IDA-specific tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {result.returncode}")
    print("=" * 70)

    return result.returncode


if __name__ == "__main__":
    try:
        # Check if we're in IDA
        import idaapi
        print("✓ Running inside IDA Pro")
        print(f"  IDA version: {idaapi.get_kernel_version()}")
        print()
    except ImportError:
        print("⚠ Warning: Not running inside IDA Pro")
        print("  Some tests will be skipped")
        print()

    sys.exit(main())
