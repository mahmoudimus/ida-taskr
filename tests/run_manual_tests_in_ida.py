"""
Run specific ida-taskr tests directly in IDA without pytest.

This script should be run with: idat -A -S"tests/run_manual_tests_in_ida.py" <binary>
"""

import sys
import os

# Add project to path
sys.path.insert(0, "/home/user/ida-taskr/src")
sys.path.insert(0, "/home/user/ida-taskr/tests")

output_file = "/tmp/ida_manual_test_results.txt"

def run_test(test_func, test_name):
    """Run a single test function and return success/failure."""
    try:
        test_func()
        return True, None
    except Exception as e:
        import traceback
        return False, traceback.format_exc()

try:
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Running IDA-specific tests manually\n")
        f.write("=" * 70 + "\n\n")

        # Verify we're in IDA
        try:
            import idaapi
            f.write(f"[+] IDA version: {idaapi.get_kernel_version()}\n")
            f.write(f"[+] sys.executable: {sys.executable}\n")
        except ImportError as e:
            f.write(f"[-] Not running in IDA: {e}\n")
            raise

        # Check is_ida()
        from ida_taskr import is_ida
        f.write(f"  is_ida() returns: {is_ida()}\n\n")

        # Check Qt application
        try:
            from PySide6.QtCore import QCoreApplication
            app = QCoreApplication.instance()
            if app:
                f.write(f"[+] Qt application: {type(app).__name__}\n\n")
            else:
                f.write("[-] No Qt application instance\n\n")
        except Exception as e:
            f.write(f"[-] Qt error: {e}\n\n")

        passed = 0
        failed = 0

        # Test 1: test_worker_launcher_integration
        f.write("-" * 70 + "\n")
        f.write("Test 1: test_worker_launcher_integration\n")
        f.write("-" * 70 + "\n")
        try:
            sys.path.insert(0, "/home/user/ida-taskr/tests/unit")
            from test_event_emitter import TestMessageEmitter
            from unittest.mock import patch

            test_instance = TestMessageEmitter()
            test_instance.setup_method(test_instance.test_worker_launcher_integration)

            with patch("ida_taskr.launcher.WorkerLauncher.launch_worker") as mock_launch_worker:
                test_instance.test_worker_launcher_integration(mock_launch_worker)

            f.write("PASSED\n\n")
            passed += 1
        except Exception as e:
            import traceback
            f.write(f"FAILED:\n{traceback.format_exc()}\n\n")
            failed += 1

        # Test 2: test_full_worker_execution
        f.write("-" * 70 + "\n")
        f.write("Test 2: test_full_worker_execution\n")
        f.write("-" * 70 + "\n")
        try:
            from test_qtasyncio import TestQtApplicationIntegration

            test_instance = TestQtApplicationIntegration()
            test_instance.test_full_worker_execution()

            f.write("PASSED\n\n")
            passed += 1
        except Exception as e:
            import traceback
            f.write(f"FAILED:\n{traceback.format_exc()}\n\n")
            failed += 1

        # Summary
        f.write("=" * 70 + "\n")
        f.write(f"Results: {passed} passed, {failed} failed\n")
        f.write("=" * 70 + "\n")

        exit_code = 0 if failed == 0 else 1

    # Print results to stdout
    with open(output_file, "r") as f:
        print(f.read())

    # Exit IDA
    try:
        import idaapi
        idaapi.qexit(exit_code)
    except:
        sys.exit(exit_code)

except Exception as e:
    with open(output_file, "a") as f:
        f.write(f"\n\nFATAL ERROR: {e}\n")
        import traceback
        traceback.print_exc(file=f)

    print(f"FATAL ERROR: {e}")

    try:
        import idaapi
        idaapi.qexit(1)
    except:
        sys.exit(1)
