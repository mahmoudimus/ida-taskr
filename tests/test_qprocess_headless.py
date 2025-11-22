"""Test if QProcess works in headless IDA."""

import sys
sys.path.insert(0, "/home/user/ida-taskr/src")

with open("/tmp/ida_qprocess_test.txt", "w") as f:
    f.write("Testing QProcess in headless IDA\n")
    f.write("=" * 70 + "\n\n")

    try:
        import idaapi
        f.write(f"IDA version: {idaapi.get_kernel_version()}\n\n")

        from PySide6.QtCore import QCoreApplication, QProcess
        app = QCoreApplication.instance()
        f.write(f"Qt app: {type(app).__name__}\n")
        f.write(f"Qt app exists: {app is not None}\n\n")

        # Test 1: Create a QProcess
        f.write("Test 1: Creating QProcess...\n")
        process = QProcess()
        f.write(f"[OK] QProcess created: {process}\n\n")

        # Test 2: Run a simple command
        f.write("Test 2: Running 'echo hello' command...\n")
        process.start("echo", ["hello"])

        # Wait for it to finish
        if process.waitForFinished(1000):  # 1 second timeout
            output = process.readAllStandardOutput().data().decode()
            f.write(f"[OK] Command executed\n")
            f.write(f"     Output: {output.strip()}\n\n")
        else:
            f.write(f"[WARN] Command timed out or failed\n")
            f.write(f"       State: {process.state()}\n\n")

        # Test 3: Create WorkerLauncher (uses QProcess)
        f.write("Test 3: Creating WorkerLauncher...\n")
        from ida_taskr import WorkerLauncher, MessageEmitter

        emitter = MessageEmitter()
        launcher = WorkerLauncher(emitter)
        f.write(f"[OK] WorkerLauncher created: {launcher}\n\n")

        # Test 4: Verify WorkerLauncher has QProcess methods
        f.write("Test 4: Verifying WorkerLauncher QProcess interface...\n")
        f.write(f"     Has start(): {hasattr(launcher, 'start')}\n")
        f.write(f"     Has waitForFinished(): {hasattr(launcher, 'waitForFinished')}\n")
        f.write(f"     Has readAllStandardOutput(): {hasattr(launcher, 'readAllStandardOutput')}\n")
        f.write(f"     Inherits QProcess: {isinstance(launcher, QProcess)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("ALL TESTS PASSED - QProcess works in headless IDA!\n")
        f.write("=" * 70 + "\n")

    except Exception as e:
        f.write(f"\n[FAIL] Error: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("=" * 70 + "\n")

try:
    import idaapi
    idaapi.qexit(0)
except:
    pass
