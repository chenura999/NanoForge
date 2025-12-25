#!/usr/bin/env python3
"""
NanoForge Python Example

Build the Python module first:
    cd nanoforge
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install maturin
    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --features python

Then run this script:
    python3 examples/demo.py
"""

try:
    import nanoforge
except ImportError:
    print("❌ NanoForge not installed!")
    print("   Build with: maturin develop --features python")
    exit(1)

print("╔══════════════════════════════════════════════════════════════╗")
print("║       🔥 NanoForge Python Demo 🔥                           ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

# 1. Check CPU features
print(f"🖥️  CPU Features: {nanoforge.cpu_features()}")

# 2. Detailed CPU info
info = nanoforge.cpu_info()
print(f"   AVX2: {info['avx2']}, AVX-512: {info['avx512f']}")

# 3. Create AI optimizer
print("\n📊 Creating AI Optimizer...")
opt = nanoforge.Optimizer()
print(f"   Variants: {opt.variant_names()}")

# 4. Simulate learning
print("\n🎰 Simulating learning iterations...")
for i in range(20):
    input_size = [10, 100, 1000, 10000][i % 4]
    variant = opt.select(input_size)
    
    # Simulate: AVX wins for large, Scalar for small
    
    if input_size >= 1000:
        cycles = 100 if variant >= 3 else 200  # AVX variants win
    else:
        cycles = 50 if variant < 3 else 150    # Scalar wins
    
    opt.update(input_size, variant, cycles, 50)
    
    if i < 5 or i == 19:
        bucket = opt.get_bucket(input_size)
        print(f"   Iter {i+1}: N={input_size:5} ({bucket}) → Variant {variant}")

# 5. Show learned decision boundary
print("\n🎯 Learned Decision Boundary:")
for bucket, variant, confidence in opt.get_decision_boundary():
    print(f"   {bucket:20} → {variant:12} (conf: {confidence:.3f})")

# 6. Save knowledge
opt.save("/tmp/nanoforge_brain.json")

# 7. Compile and run a simple script
print("\n⚙️  Compiling NanoForge script...")
try:
    func = nanoforge.compile("""
        x = 42
        y = x + 10
        return y
    """)
    result = func(0)  # Input not used in this script
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Compile error: {e}")

print(f"\n✅ NanoForge version: {nanoforge.version()}")
