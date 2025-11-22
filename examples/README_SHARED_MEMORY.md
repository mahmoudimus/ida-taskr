# Minimal Shared Memory API

## Question: What's the smallest surface area for shared memory tasks?

**Answer: Just write the chunk processing logic!**

---

## The Decorator

```python
@shared_memory_task(num_chunks=8)
def process_chunk(chunk_data, chunk_id, total_chunks):
    # Your chunk processing logic
    return result
```

That's it. **User writes ~5 lines**. ida-taskr handles ~40 lines of boilerplate!

---

## What User Provides

**Just the chunk processing function:**

```python
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze_chunk(chunk_data, chunk_id, total_chunks):
    """
    Process one chunk of data.

    Args:
        chunk_data: memoryview of this chunk (no copying!)
        chunk_id: 0-based chunk index (0 to 7)
        total_chunks: total number of chunks (8)

    Returns:
        Results from processing this chunk
    """
    signatures = []

    for i in range(len(chunk_data) - 16):
        if chunk_data[i] == 0x48:  # Pattern
            sig = bytes(chunk_data[i:i+16])
            signatures.append(sig.hex())

    return signatures
```

**Usage - just pass the full data:**

```python
# Get binary data (e.g., from IDA)
binary_data = get_binary_data()  # 8MB

# Process it - returns list of results from all 8 chunks
all_results = analyze_chunk(binary_data)

# That's it!
```

**Total user code: ~10 lines**

---

## What ida-taskr Handles

When you use `@shared_memory_task`, ida-taskr automatically:

1. **Creates shared memory** segment
2. **Copies data once** into shared memory (no per-chunk copying!)
3. **Calculates chunk boundaries** (splits data into N equal chunks)
4. **Creates worker processes** (ProcessPoolExecutor with N workers)
5. **Attaches workers to shared memory** (via shm.name, no data transfer!)
6. **Submits all chunks** with proper offsets
7. **Collects results** from all workers
8. **Cleanup memoryview** before closing (CRITICAL!)
9. **Cleanup shared memory** (close and unlink)

**Total boilerplate handled: ~40 lines**

---

## Comparison

### Without Decorator (Manual Setup)

```python
def analyze_manual(binary_data):
    import multiprocessing.shared_memory
    from ida_taskr import ProcessPoolExecutor

    # 1. Create shared memory
    shm = multiprocessing.shared_memory.SharedMemory(
        create=True, size=len(binary_data)
    )

    try:
        # 2. Copy data
        shm.buf[:] = binary_data

        # 3. Calculate chunks
        num_chunks = 8
        chunk_size = len(binary_data) // num_chunks

        # 4. Define worker with attachment logic
        def worker(shm_name, start, end, chunk_id):
            shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
            try:
                chunk_data = memoryview(shm.buf)[start:end]
                # ... process chunk ...
                return result
            finally:
                del chunk_data  # CRITICAL!
                shm.close()

        # 5. Submit chunks
        executor = ProcessPoolExecutor(max_workers=8)
        futures = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(binary_data)
            future = executor.submit(worker, shm.name, start, end, i)
            futures.append(future)

        # 6. Collect results
        results = [f.result() for f in futures]

        # 7. Cleanup executor
        executor.shutdown()

    finally:
        # 8. Cleanup shared memory
        shm.close()
        shm.unlink()

    return results

# ~40-50 lines of boilerplate!
```

### With Decorator (MINIMAL)

```python
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=8)
def analyze_minimal(chunk_data, chunk_id, total_chunks):
    # ... process chunk ...
    return result

# Usage
results = analyze_minimal(binary_data)

# ~5-10 lines total!
```

---

## Real-World IDA Example

```python
import ida_bytes
from ida_taskr import shared_memory_task

@shared_memory_task(num_chunks=16)
def find_function_prologues(chunk_data, chunk_id, total_chunks):
    """Find x64 function prologues in this chunk."""
    patterns = []

    # Common prologues
    prologue = b'\x55\x48\x89\xe5'  # push rbp; mov rbp, rsp

    for i in range(len(chunk_data) - 4):
        if chunk_data[i:i+4] == prologue:
            patterns.append({
                'chunk': chunk_id,
                'offset': i,
                'pattern': 'push_rbp_mov_rbp_rsp'
            })

    return patterns


def analyze_current_segment():
    """Analyze the current IDA segment."""
    # Get binary data from IDA
    start_ea = idc.get_segm_start(idc.here())
    end_ea = idc.get_segm_end(idc.here())
    binary_data = ida_bytes.get_bytes(start_ea, end_ea - start_ea)

    print(f"Analyzing {len(binary_data):,} bytes in 16 parallel chunks...")

    # Process in parallel - just one line!
    all_patterns = find_function_prologues(binary_data)

    # Show results
    total = sum(len(chunk) for chunk in all_patterns)
    print(f"Found {total} function prologues!")

    return all_patterns
```

**User writes:**
- Chunk processing logic (~10 lines)
- Usage code (~3 lines)

**ida-taskr handles:**
- All shared memory complexity (~40 lines)

---

## Why Shared Memory?

**Problem:** IDA takes locks on main thread. Can't call IDA SDK from workers.

**Solution:** Copy data from IDA once, then workers never touch IDA!

```python
# In IDA's main thread (has the locks)
binary_data = ida_bytes.get_bytes(start_ea, size)  # Get data ONCE

# Workers process shared memory (no IDA SDK calls needed!)
@shared_memory_task(num_chunks=8)
def analyze(chunk_data, chunk_id, total_chunks):
    # No IDA SDK calls here!
    # Just process the raw bytes
    return find_patterns(chunk_data)

# True parallel execution across all CPU cores
results = analyze(binary_data)
```

**Benefits:**
- ✓ Copy data ONCE (not per chunk)
- ✓ No serialization overhead
- ✓ True parallel processing
- ✓ Workers never touch IDA (no lock issues)
- ✓ Can process GB of data efficiently

---

## Summary

| Aspect | Manual | With Decorator |
|--------|--------|----------------|
| **User code** | ~40-50 lines | ~5-10 lines |
| **Boilerplate** | All manual | Automated |
| **SharedMemory** | Manual create/cleanup | Automatic |
| **Chunking** | Manual math | Automatic |
| **Worker attachment** | Manual | Automatic |
| **Cleanup** | Manual (easy to forget!) | Automatic |

---

## The Answer

**Smallest surface area for shared memory tasks:**

```python
@shared_memory_task(num_chunks=N)
def process_chunk(chunk_data, chunk_id, total_chunks):
    # Your logic here
    return result
```

**User writes:** Just the chunk processing logic (~5-10 lines)

**ida-taskr handles:** Everything else (~40 lines of boilerplate)

---

## Examples in This Directory

- **`shared_memory_one_line.py`** - Absolute minimal example
- **`shared_memory_minimal.py`** - Before/after comparison
- **`shared_memory_comparison.py`** - Detailed comparison with IDA examples
- **`shared_memory_decorator_example.py`** - Working example with real data

Start with `shared_memory_one_line.py` to see the smallest possible usage.
