import sys
sys.path.insert(0, "/home/user/ida-taskr/src")
sys.path.insert(0, "/home/user/ida-taskr/tests/unit")

with open("/tmp/ida_launcher_test.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("Running test_worker_launcher_integration in IDA\n")
    f.write("=" * 70 + "\n\n")

    try:
        import idaapi
        f.write(f"IDA version: {idaapi.get_kernel_version()}\n\n")

        from unittest.mock import patch
        from test_event_emitter import TestMessageEmitter

        f.write("Creating test instance...\n")
        test_instance = TestMessageEmitter()
        test_instance.setup_method(test_instance.test_worker_launcher_integration)
        f.write("[OK] Test instance created\n\n")

        f.write("Running test with mocked launch_worker...\n")
        with patch("ida_taskr.launcher.WorkerLauncher.launch_worker") as mock_launch_worker:
            mock_launch_worker.return_value = True
            test_instance.test_worker_launcher_integration(mock_launch_worker)

        f.write("[OK] TEST PASSED!\n")
        f.write("=" * 70 + "\n")

    except Exception as e:
        f.write(f"\n[FAIL] Test failed: {e}\n")
        import traceback
        f.write(traceback.format_exc())
        f.write("=" * 70 + "\n")

try:
    import idaapi
    idaapi.qexit(0)
except:
    pass
