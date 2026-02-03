import sys
sys.path.insert(0, "/home/user/ida-taskr/src")

with open("/tmp/ida_imports_test.txt", "w") as f:
    f.write("Testing imports...\n\n")

    try:
        f.write("1. Importing ida_taskr...\n")
        import ida_taskr
        f.write("   [OK] Success\n\n")

        f.write("2. Checking is_ida()...\n")
        f.write(f"   Result: {ida_taskr.is_ida()}\n\n")

        f.write("3. Importing MessageEmitter...\n")
        from ida_taskr import MessageEmitter
        f.write("   [OK] Success\n\n")

        f.write("4. Creating MessageEmitter...\n")
        emitter = MessageEmitter()
        f.write(f"   [OK] Created: {emitter}\n\n")

        f.write("5. Importing WorkerLauncher...\n")
        from ida_taskr import WorkerLauncher
        f.write("   [OK] Success\n\n")

        f.write("6. Creating WorkerLauncher...\n")
        launcher = WorkerLauncher(emitter)
        f.write(f"   [OK] Created: {launcher}\n\n")

        f.write("ALL TESTS PASSED!\n")

    except Exception as e:
        f.write(f"\nError: {e}\n")
        import traceback
        f.write(traceback.format_exc())

try:
    import idaapi
    idaapi.qexit(0)
except:
    pass
