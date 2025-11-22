with open("/tmp/ida_minimal.txt", "w") as f:
    f.write("Script started\n")

try:
    import idaapi
    with open("/tmp/ida_minimal.txt", "a") as f:
        f.write(f"IDA version: {idaapi.get_kernel_version()}\n")
    idaapi.qexit(0)
except Exception as e:
    with open("/tmp/ida_minimal.txt", "a") as f:
        f.write(f"Error: {e}\n")
