# NanoForge
**A JIT compiler that evolves. NanoForge rewrites its own assembly at runtime to match your exact CPU hardware.**

![NanoForge Demo](https://via.placeholder.com/800x400?text=Hero+Shot+Pending)

> "Optimizes faster than you can blink."

## 🚀 The "Hero" Shot
NanoForge isn't just a compiler; it's a living runtime. Watch it upgrade a running function from Scalar logic to AVX2 Vector instructions *without stopping execution*.

```bash
$ cargo run --release --bin nanoforge -- adaptive vec_add_stress.nf

=== NanoForge Adaptive Runtime ===
Running Tier 1 (Scalar)... 1.68 cycles/op

🔥 HOT SWAP TRIGGERED 🔥

Running Tier 2 (AVX2)... 1.01 cycles/op

Speedup: 1.65x
```

## ⚡ Benchmarks
Performance scaling on a typical workload (Vector Addition):

```
Cycles per Element (Lower is Better)
┌─────────────────────────────────┐
│ Scalar (No Opt)     : 16.0 cyc  │ 
├─────────────────────────────────┤
│ Tier 1 (Unrolled)   : 1.68 cyc  │ 
├─────────────────────────────────┤
│ Tier 2 (AVX2 SIMD)  : 1.01 cyc  │ 🚀 (Instruction Limit!)
└─────────────────────────────────┘
```

## ✨ Features
- **Adaptive Compilation**: Starts with fast-compiling Tier 1 code, then Hot-Swaps to Tier 2 (AVX2) when load is detected.
- **Auto-Vectorization**: Automatically detects `Load-Add-Store` loops and transforms them into 256-bit AVX2 SIMD chains.
- **Safety Valve**: Handling remainder elements (loop count % 4 != 0) with a scalar cleanup loop.
- **Pure Rust Backend**: Custom `x64` assembler and `dynasm` implementation—no LLVM dependency.

## 🛠️ Quick Start

### Prerequisites
- Linux/x64
- Rust Toolchain
- AVX2 supported CPU

### Run the Demo
```bash
# Clone
git clone https://github.com/alonexe/NanoForge
cd NanoForge/nanoforge

# Run Adaptive Demo
cargo run --release --bin nanoforge -- adaptive vec_add_stress.nf
```

## 🏗️ Architecture
1. **Parser**: Generates NanoForge IR.
2. **Compiler**: Two-Tier strategy.
   - **Tier 1**: Unrolled Scalar loops (Optimization Level 2).
   - **Tier 2**: Vectorized AVX2 loops (Optimization Level 3).
3. **Runtime**: Monitors function execution count and triggers **Hot Swap** via function pointer replacement.

---
Built with ❤️ and Rust.
