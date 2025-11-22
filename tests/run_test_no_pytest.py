"""Run test_worker_launcher_integration without pytest."""

import sys
sys.path.insert(0, "/home/user/ida-taskr/src")

with open("/tmp/ida_test_no_pytest.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("Running test_worker_launcher_integration in IDA (no pytest)\n")
    f.write("=" * 70 + "\n\n")

    try:
        import idaapi
        f.write(f"IDA version: {idaapi.get_kernel_version()}\n\n")

        # Import what we need
        from unittest.mock import patch
        from ida_taskr import MessageEmitter, WorkerLauncher

        # Create test class manually (without pytest)
        class TestMessageEmitter:
            def setup_method(self, method):
                self.emitter = MessageEmitter()
                self.test_events = []

            def test_worker_launcher_integration(self, mock_launch_worker):
                # Configure the mock
                mock_launch_worker.return_value = True

                # Create message emitter with event handlers
                message_emitter = MessageEmitter()
                connection_events = []

                @message_emitter.on("worker_connected")
                def on_connected():
                    connection_events.append("connected")

                @message_emitter.on("worker_disconnected")
                def on_disconnected():
                    connection_events.append("disconnected")

                # Create worker launcher with the message emitter
                launcher = WorkerLauncher(message_emitter)

                # Launch worker
                worker_args = {"data_size": 1024, "start_ea": "0x1000", "is64": "1"}
                result = launcher.launch_worker("path/to/test/worker.py", worker_args)

                # Verify
                assert result == True, f"Expected True, got {result}"
                assert mock_launch_worker.called, "launch_worker should have been called"

                f.write("[OK] All assertions passed\n")

        # Run the test
        f.write("Creating test instance...\n")
        test_instance = TestMessageEmitter()
        test_instance.setup_method(test_instance.test_worker_launcher_integration)
        f.write("[OK] Setup complete\n\n")

        f.write("Running test with mocked launch_worker...\n")
        with patch("ida_taskr.launcher.WorkerLauncher.launch_worker") as mock_launch_worker:
            mock_launch_worker.return_value = True
            test_instance.test_worker_launcher_integration(mock_launch_worker)

        f.write("\n[OK] TEST PASSED!\n")
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
