"""Script to run test plugin in IDA and exit."""

import idaapi
import sys

# Wait for auto-analysis to complete
idaapi.auto_wait()

# Load and run the plugin
print("\nLoading test_ida_taskr plugin...")
plugin_name = "Test ida-taskr"

# Run the plugin
idaapi.load_plugin("test_ida_taskr")
idaapi.run_plugin(idaapi.find_plugin("test_ida_taskr", True), 0)

# Exit IDA
print("\nExiting IDA...")
idaapi.qexit(0)
