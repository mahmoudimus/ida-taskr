"""Run test_worker_launcher_integration directly in IDA."""

import sys
import os

# Add paths
sys.path.insert(0, "/home/user/ida-taskr/src")
sys.path.insert(0, "/home/user/ida-taskr/tests/unit")

output_file = "/tmp/ida_single_test.txt"

try:
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Running test_worker_launcher_integration in IDA Pro\n")
        f.write("=" * 70 + "\n\n")

        # Check environment
        try:
            import idaapi
            f.write(f"[+] IDA version: {idaapi.get_kernel_version()}\n")
        except ImportError as e:
            f.write(f"[-] Not in IDA: {e}\n")
            raise

        from ida_taskr import is_ida
        f.write(f"[+] is_ida() returns: {is_ida()}\n")

        # Check Qt
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app:
            f.write(f"[+] Qt app: {type(app).__name__}\n\n")
        else:
            f.write("[-] No Qt app\n\n")

        # Run the test
        f.write("-" * 70 + "\n")
        f.write("Running test...\n")
        f.write("-" * 70 + "\n")

        from unittest.mock import patch
        from test_event_emitter import TestMessageEmitter

        test_instance = TestMessageEmitter()
        test_instance.setup_method(test_instance.test_worker_launcher_integration)

        with patch("ida_taskr.launcher.WorkerLauncher.launch_worker") as mock_launch_worker:
            mock_launch_worker.return_value = True
            test_instance.test_worker_launcher_integration(mock_launch_worker)

        f.write("\n✓ TEST PASSED!\n")
        f.write("=" * 70 + "\n")

    # Print to stdout too
    with open(output_file, "r") as f:
        print(f.read())

except Exception as e:
    with open(output_file, "a") as f:
        f.write(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.write("=" * 70 + "\n")

    with open(output_file, "r") as f:
        print(f.read())

# Exit IDA
try:
    import idaapi
    idaapi.qexit(0)
except:
    sys.exit(0)
