"""Simple check of IDA environment."""

import sys

output = []

try:
    output.append("Step 1: Import idaapi")
    import idaapi
    output.append(f"  IDA version: {idaapi.get_kernel_version()}")

    output.append("\nStep 2: Check Qt")
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance()
    output.append(f"  Qt app: {type(app).__name__ if app else 'None'}")

    output.append("\nStep 3: Add paths")
    sys.path.insert(0, "/home/user/ida-taskr/src")
    output.append("  Paths added")

    output.append("\nStep 4: Import ida_taskr")
    import ida_taskr
    output.append(f"  is_ida(): {ida_taskr.is_ida()}")

    output.append("\nStep 5: Import MessageEmitter")
    from ida_taskr import MessageEmitter
    output.append("  MessageEmitter imported")

    output.append("\nStep 6: Create MessageEmitter")
    emitter = MessageEmitter()
    output.append(f"  Created: {emitter}")

    output.append("\nStep 7: Import WorkerLauncher")
    from ida_taskr import WorkerLauncher
    output.append("  WorkerLauncher imported")

    output.append("\nStep 8: Create WorkerLauncher")
    launcher = WorkerLauncher(emitter)
    output.append(f"  Created: {launcher}")

    output.append("\n✓ ALL STEPS COMPLETED")

except Exception as e:
    output.append(f"\n✗ Error at current step: {e}")
    import traceback
    output.append(traceback.format_exc())

# Write output
with open("/tmp/ida_simple_check.txt", "w") as f:
    f.write("\n".join(output))

# Print output
print("\n".join(output))

# Exit
try:
    import idaapi
    idaapi.qexit(0)
except:
    pass
