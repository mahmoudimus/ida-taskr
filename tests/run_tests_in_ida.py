"""
Run ida-taskr tests directly inside IDA's Python environment.

This script should be run with: idat -A -S"tests/run_tests_in_ida.py" <binary>
"""

import sys
import os

# Add project to path
sys.path.insert(0, "/home/user/ida-taskr/src")
sys.path.insert(0, "/home/user/ida-taskr")

# Set environment variables pytest might need
os.environ['PYTHONPATH'] = "/home/user/ida-taskr/src"

output_file = "/tmp/ida_test_results.txt"

try:
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Running tests inside IDA Pro\n")
        f.write("=" * 70 + "\n\n")

        # Verify we're in IDA
        try:
            import idaapi
            f.write(f"✓ IDA version: {idaapi.get_kernel_version()}\n")
            f.write(f"✓ sys.executable: {sys.executable}\n")
        except ImportError:
            f.write("✗ Not running in IDA!\n")
            sys.exit(1)

        # Check is_ida()
        from ida_taskr import is_ida
        f.write(f"✓ is_ida() returns: {is_ida()}\n\n")

        # Import pytest and run tests
        f.write("Running pytest...\n")
        f.flush()

    # Run pytest in-process
    import pytest

    # Run the two IDA-specific tests
    exit_code = pytest.main([
        "tests/unit/test_event_emitter.py::TestMessageEmitter::test_worker_launcher_integration",
        "tests/unit/test_qtasyncio.py::TestQtApplicationIntegration::test_full_worker_execution",
        "-v",
        "--tb=short",
        "-s",  # Don't capture output
    ])

    with open(output_file, "a") as f:
        f.write(f"\n\nTests completed with exit code: {exit_code}\n")

    # Exit IDA
    try:
        import idaapi
        idaapi.qexit(exit_code)
    except:
        sys.exit(exit_code)

except Exception as e:
    with open(output_file, "a") as f:
        f.write(f"\n\nERROR: {e}\n")
        import traceback
        traceback.print_exc(file=f)

    try:
        import idaapi
        idaapi.qexit(1)
    except:
        sys.exit(1)
