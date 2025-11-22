"""Check if WorkerLauncher can be instantiated in IDA."""

import sys
sys.path.insert(0, "/home/user/ida-taskr/src")

output_file = "/tmp/ida_launcher_check.txt"

try:
    with open(output_file, "w") as f:
        import idaapi
        f.write(f"IDA version: {idaapi.get_kernel_version()}\n")

        from PySide6.QtCore import QCoreApplication, QApplication
        app = QCoreApplication.instance()
        f.write(f"Qt app type: {type(app).__name__}\n")
        f.write(f"Qt app: {app}\n\n")

        # Try to import WorkerLauncher
        f.write("Importing ida_taskr...\n")
        from ida_taskr import WorkerLauncher, MessageEmitter
        f.write("✓ Import successful\n\n")

        # Try to create MessageEmitter
        f.write("Creating MessageEmitter...\n")
        emitter = MessageEmitter()
        f.write("✓ MessageEmitter created\n\n")

        # Try to create WorkerLauncher
        f.write("Creating WorkerLauncher...\n")
        launcher = WorkerLauncher(emitter)
        f.write(f"✓ WorkerLauncher created: {launcher}\n")

except Exception as e:
    with open(output_file, "a") as f:
        f.write(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)

with open(output_file, "r") as f:
    print(f.read())

try:
    import idaapi
    idaapi.qexit(0)
except:
    sys.exit(0)
