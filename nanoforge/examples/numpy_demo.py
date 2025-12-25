#!/usr/bin/env python3
"""
NanoForge NumPy Demo - AVX2 Accelerated Array Operations

Build the Python module first:
    cd nanoforge
    maturin develop --features python

Then run this script:
    python3 examples/numpy_demo.py
"""

import sys

try:
    import numpy as np
    import nanoforge
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Build with: maturin develop --features python")
    sys.exit(1)


def format_ns(ns):
    """Format nanoseconds nicely"""
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.2f} s"
    elif ns >= 1_000_000:
        return f"{ns / 1e6:.2f} ms"
    elif ns >= 1_000:
        return f"{ns / 1e3:.2f} µs"
    else:
        return f"{ns:.0f} ns"


def is_aligned(arr, alignment=32):
    """Check if array is aligned to given byte boundary"""
    return arr.ctypes.data % alignment == 0


def create_aligned_array(size, alignment=32, dtype=np.int64):
    """Create an array that is guaranteed to be aligned to given byte boundary"""
    # Allocate extra space and find aligned start
    itemsize = np.dtype(dtype).itemsize
    extra = alignment // itemsize
    arr = np.zeros(size + extra, dtype=dtype)
    offset = (alignment - (arr.ctypes.data % alignment)) % alignment // itemsize
    return arr[offset : offset + size]


print("╔══════════════════════════════════════════════════════════════╗")
print("║     🔥 NanoForge NumPy Demo - AVX2 Acceleration 🔥          ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# 1. Check CPU features
print(f"🖥️  CPU Features: {nanoforge.cpu_features()}")
info = nanoforge.cpu_info()
print(f"   AVX2: {info['avx2']}, AVX-512: {info['avx512f']}\n")

# 2. Basic functionality test
print("📋 Testing vec_add correctness...")
a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
b = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
c = np.zeros(8, dtype=np.int64)

nanoforge.vec_add(a, b, c)
expected = a + b
assert np.array_equal(c, expected), f"Mismatch: {c} != {expected}"
print(f"   ✅ vec_add: {a} + {b} = {c}")

# 3. Test vec_sum
print("\n📋 Testing vec_sum correctness...")
arr = np.arange(100, dtype=np.int64)
total = nanoforge.vec_sum(arr)
expected_sum = int(arr.sum())
assert total == expected_sum, f"Sum mismatch: {total} != {expected_sum}"
print(f"   ✅ vec_sum: sum(0..99) = {total}")

# 4. Test vec_scale
print("\n📋 Testing vec_scale correctness...")
arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
nanoforge.vec_scale(arr, 10)
expected_scaled = np.array([10, 20, 30, 40, 50], dtype=np.int64)
assert np.array_equal(arr, expected_scaled), (
    f"Scale mismatch: {arr} != {expected_scaled}"
)
print(f"   ✅ vec_scale: arr *= 10 = {arr}")

# 5. Performance benchmark
print("\n" + "=" * 64)
print("🚀 PERFORMANCE BENCHMARK: NanoForge vs NumPy")
print("=" * 64)

import time

for size in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
    # Create arrays
    a = np.arange(size, dtype=np.int64)
    b = np.arange(size, dtype=np.int64) * 2
    c = np.zeros(size, dtype=np.int64)

    iterations = 100 if size < 1_000_000 else 10

    # Warmup
    nanoforge.vec_add(a, b, c)
    np.add(a, b, out=c)

    # Benchmark NanoForge
    start = time.perf_counter_ns()
    for _ in range(iterations):
        nanoforge.vec_add(a, b, c)
    nanoforge_ns = (time.perf_counter_ns() - start) // iterations

    # Benchmark NumPy
    start = time.perf_counter_ns()
    for _ in range(iterations):
        np.add(a, b, out=c)
    numpy_ns = (time.perf_counter_ns() - start) // iterations

    # Calculate speedup
    if nanoforge_ns > 0:
        speedup = numpy_ns / nanoforge_ns
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1:
            speedup_str = f"✅ {speedup_str}"
        else:
            speedup_str = f"❌ {speedup_str}"
    else:
        speedup_str = "∞"

    aligned = "🎯" if is_aligned(c) else ""
    print(f"\n   N = {size:>10,} {aligned}")
    print(f"   NanoForge: {format_ns(nanoforge_ns):>10}")
    print(f"   NumPy:     {format_ns(numpy_ns):>10}")
    print(f"   Speedup:   {speedup_str}")

# 6. Large scale test with FORCED 32-byte alignment for NT stores
print("\n" + "=" * 64)
print("🧪 LARGE SCALE TEST: 100M elements (32-byte aligned for NT stores)")
print("=" * 64)

size = 100_000_000
print(f"   Allocating 3 x {size:,} int64 arrays ({size * 8 * 3 / 1e9:.1f} GB)...")

try:
    # Force 32-byte alignment for NT store path
    a = create_aligned_array(size)
    b = create_aligned_array(size)
    c = create_aligned_array(size)

    # Initialize
    a[:] = np.arange(size, dtype=np.int64)
    b[:] = a * 2
    c[:] = 0

    print(
        f"   Alignment: a={is_aligned(a)}, b={is_aligned(b)}, c={is_aligned(c)} (32-byte)"
    )

    # NanoForge with NT stores
    start = time.perf_counter_ns()
    nanoforge.vec_add(a, b, c)
    nanoforge_ns = time.perf_counter_ns() - start

    # NumPy baseline
    c[:] = 0  # Reset
    start = time.perf_counter_ns()
    np.add(a, b, out=c)
    numpy_ns = time.perf_counter_ns() - start

    speedup = numpy_ns / nanoforge_ns if nanoforge_ns > 0 else 0

    print(f"   NanoForge: {format_ns(nanoforge_ns)} (NT stores enabled)")
    print(f"   NumPy:     {format_ns(numpy_ns)}")
    if speedup >= 1.0:
        print(f"   Speedup:   ✅ {speedup:.2f}x")
    else:
        print(f"   Speedup:   {speedup:.2f}x")

    # Verify correctness
    expected = a + b
    if np.array_equal(c, expected):
        print("   ✅ Result verified correct!")
    else:
        print("   ❌ Result mismatch!")

except MemoryError:
    print("   ⚠️ Not enough memory for 100M test, skipping...")

print(f"\n✅ NanoForge version: {nanoforge.version()}")
print("🎉 All tests passed!")
