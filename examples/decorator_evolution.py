"""
Example: Evolution from simple function to decorated async task

Shows how decorators can simplify the API step by step.
"""

import time


# ==============================================================================
# STEP 1: Just the function (no async, blocks main thread)
# ==============================================================================

def generate_signature_v1(address, data, max_length=32):
    """Simple function - blocks the caller."""
    signature = []

    for i in range(min(max_length, len(data))):
        signature.append(data[i])
        time.sleep(0.01)  # Simulate CPU work

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# result = generate_signature_v1(0x401000, data)  # ← BLOCKS for entire duration!
# print(result['pattern'])


# ==============================================================================
# STEP 2: Add @background_task decorator (simplest async)
# ==============================================================================

from ida_taskr import background_task

@background_task
def generate_signature_v2(address, data, max_length=32):
    """
    Same function, but with @background_task decorator.

    Now runs in background process automatically!
    """
    signature = []

    for i in range(min(max_length, len(data))):
        signature.append(data[i])
        time.sleep(0.01)

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# future = generate_signature_v2(0x401000, data)  # ← Returns immediately!
# result = future.result()  # Wait for result when needed
# print(result['pattern'])


# ==============================================================================
# STEP 3: Add callback for when task completes
# ==============================================================================

@background_task(on_complete=lambda result: print(f"Done: {result['pattern']}"))
def generate_signature_v3(address, data, max_length=32):
    """With automatic callback on completion."""
    signature = []

    for i in range(min(max_length, len(data))):
        signature.append(data[i])
        time.sleep(0.01)

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# generate_signature_v3(0x401000, data)  # ← Returns immediately, calls callback when done!


# ==============================================================================
# STEP 4: Add progress reporting
# ==============================================================================

@background_task(
    on_progress=lambda progress, msg: print(f"[{progress}%] {msg}"),
    on_complete=lambda result: print(f"✓ Done: {result['pattern']}")
)
def generate_signature_v4(address, data, max_length=32, progress_callback=None):
    """With progress reporting."""
    signature = []
    total = min(max_length, len(data))

    for i in range(total):
        signature.append(data[i])
        time.sleep(0.01)

        # Report progress
        if progress_callback and i % 4 == 0:
            progress = int((i + 1) / total * 100)
            progress_callback(progress, f"Processing byte {i+1}/{total}")

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# generate_signature_v4(0x401000, data)  # ← Progress updates appear automatically!


# ==============================================================================
# STEP 5: Multiple concurrent tasks
# ==============================================================================

@background_task(max_workers=4, on_complete=lambda r: print(f"✓ 0x{r['address']:X}"))
def generate_signature_v5(address, data, max_length=32):
    """Can run multiple in parallel."""
    signature = []

    for i in range(min(max_length, len(data))):
        signature.append(data[i])
        time.sleep(0.01)

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# # Submit multiple - all run in parallel!
# f1 = generate_signature_v5(0x401000, data1)
# f2 = generate_signature_v5(0x402000, data2)
# f3 = generate_signature_v5(0x403000, data3)
# f4 = generate_signature_v5(0x404000, data4)


# ==============================================================================
# STEP 6: Full featured with error handling
# ==============================================================================

@background_task(
    max_workers=4,
    on_complete=lambda r: print(f"✓ Signature: {r['pattern']}"),
    on_error=lambda e: print(f"✗ Error: {e}"),
    on_progress=lambda p, m: print(f"[{p}%] {m}"),
    timeout=60.0  # Optional timeout
)
def generate_signature_v6(address, data, max_length=32, progress_callback=None):
    """Full featured with all callbacks."""
    signature = []
    total = min(max_length, len(data))

    for i in range(total):
        signature.append(data[i])
        time.sleep(0.01)

        if progress_callback and i % 4 == 0:
            progress = int((i + 1) / total * 100)
            progress_callback(progress, f"Analyzing byte {i+1}/{total}")

    return {
        'address': address,
        'pattern': ' '.join(f'{b:02X}' for b in signature[:16])
    }

# Usage:
# generate_signature_v6(0x401000, data)  # ← Everything handled automatically!


# ==============================================================================
# DEMO: Compare all versions
# ==============================================================================

def demo_evolution():
    """Show the evolution from blocking to async with decorators."""

    data = bytes(range(256))

    print("=" * 70)
    print("API Evolution: From Blocking to Background Tasks")
    print("=" * 70)
    print()

    # V1: Blocking
    print("V1: Blocking function (old way)")
    print("-" * 70)
    print("result = generate_signature_v1(address, data)")
    print("  ↑ Blocks for entire duration")
    print("  ↑ UI freezes")
    print()

    # V2: Background task
    print("V2: @background_task (simplest async)")
    print("-" * 70)
    print("@background_task")
    print("def generate_signature_v2(...):")
    print()
    print("future = generate_signature_v2(address, data)")
    print("  ↑ Returns immediately!")
    print("  ↑ UI stays responsive")
    print()

    # V3: With callback
    print("V3: Auto-callback on completion")
    print("-" * 70)
    print("@background_task(on_complete=lambda r: print(r))")
    print("def generate_signature_v3(...):")
    print()
    print("generate_signature_v3(address, data)")
    print("  ↑ Result delivered automatically via callback")
    print()

    # V4: With progress
    print("V4: Progress reporting")
    print("-" * 70)
    print("@background_task(")
    print("    on_progress=lambda p, m: update_ui(p, m),")
    print("    on_complete=lambda r: show_result(r)")
    print(")")
    print("def generate_signature_v4(...):")
    print("  ↑ Real-time progress updates")
    print()

    # V5: Parallel
    print("V5: Parallel execution")
    print("-" * 70)
    print("@background_task(max_workers=4)")
    print("def generate_signature_v5(...):")
    print()
    print("# All run in parallel:")
    print("f1 = generate_signature_v5(addr1, data1)")
    print("f2 = generate_signature_v5(addr2, data2)")
    print("f3 = generate_signature_v5(addr3, data3)")
    print("  ↑ Uses 4 CPU cores in parallel")
    print()

    # V6: Full featured
    print("V6: Full featured")
    print("-" * 70)
    print("@background_task(")
    print("    max_workers=4,")
    print("    on_complete=show_result,")
    print("    on_error=show_error,")
    print("    on_progress=update_ui,")
    print("    timeout=60.0")
    print(")")
    print("def generate_signature_v6(...):")
    print("  ↑ Everything handled automatically!")
    print()

    print("=" * 70)
    print("From 1 line change (@background_task) to full async!")
    print("=" * 70)


if __name__ == '__main__':
    demo_evolution()
