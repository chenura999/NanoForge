# NanoForge 🔥
**A Self-Optimizing Assembly Engine (SOAE) with AI-Powered Variant Selection**

NanoForge doesn't just compile—it *learns*. Using Thompson Sampling and Contextual Bandits, it dynamically selects the fastest code variant for your exact CPU and workload.

## 🧠 What Makes It Special

```bash
$ cargo run --release --bin nanoforge -- soae-context matmul_stress.nf -i 100

🧠 CONTEXTUAL BANDIT - Learning Decision Boundaries!

🎯 Learned Decision Boundary:
┌──────────────────┬──────────────────┬───────────┐
│ Input Size       │ Best Variant     │ Confidence│
├──────────────────┼──────────────────┼───────────┤
│ Tiny (<32)       │ Scalarx16        │     0.603 │  ← Scalar wins for Tiny!
│ Small (32-255)   │ Scalarx2         │     0.623 │  ← Scalar wins for Small!
│ Medium (256-4K)  │ AVX2x2           │     0.640 │  ← AVX2 wins for Medium!
│ Large (4K-64K)   │ AVX2x4           │     0.600 │  ← AVX2 wins for Large!
│ Huge (>64K)      │ Scalarx16        │     0.616 │  ← Scalarx16 (Spilled) wins for Huge!
└──────────────────┴──────────────────┴───────────┘
```

**The AI learns: Small inputs → Scalar, Large inputs → AVX2**

## ⚡ Features

| Feature | Description |
|---------|-------------|
| **Multi-Variant Generation** | Generates 6+ optimized variants per function (Scalar, AVX2) |
| **Nanosecond Sandbox** | RDTSC cycle-accurate benchmarking |
| **Thompson Sampling** | Bayesian bandit for exploration/exploitation |
| **Contextual Learning** | Learns different policies for different input sizes |
| **Hot Swap** | Replaces running code without stopping execution |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/alonexe/NanoForge
cd NanoForge/nanoforge

# Run AI-Powered SOAE Demo
cargo run --release --bin nanoforge -- soae-context vec_add_stress.nf -i 100
```

## 📊 Demo Commands

| Command | Description |
|---------|-------------|
| `soae <file>` | Benchmark all variants, pick winner |
| `soae-ai <file> -i N` | Thompson Sampling learning (N iterations) |
| `soae-context <file> -i N` | **Contextual learning with decision boundaries** |
| `adaptive <file>` | Classic hot-swap tier demo |

## 🏗️ Architecture

```
┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
│   Parser    │───▶│  Compiler     │───▶│ Variant Generator│
└─────────────┘    └───────────────┘    └────────┬────────┘
                                                 │
                   ┌─────────────────────────────▼────────┐
                   │     Nanosecond Sandbox (RDTSC)       │
                   └─────────────────────────────┬────────┘
                                                 │
                   ┌─────────────────────────────▼────────┐
                   │  Contextual Bandit (Thompson Sampling)│
                   │  - Learns per-bucket policies         │
                   │  - Selects optimal variant            │
                   └──────────────────────────────────────┘
```

## 📁 Key Modules

| Module | Purpose |
|--------|---------|
| `ai_optimizer.rs` | Thompson Sampling, Contextual Bandit, SizeBucket |
| `variant_generator.rs` | Multi-variant code generation |
| `sandbox.rs` | RDTSC cycle-accurate benchmarking |
| `cpu_features.rs` | CPUID-based ISA detection |

## 📈 Performance

```
Cycles per Operation (Lower is Better)
┌─────────────────────────────────┐
│ Scalar (Tier 1)   : 1.68 cyc/op │
├─────────────────────────────────┤
│ AVX2 (Tier 2)     : 1.01 cyc/op │ 🚀 1.69x faster!
└─────────────────────────────────┘
```

## 🔧 Requirements

- Linux x86_64
- Rust 1.70+
- AVX2 CPU (Intel Haswell+ / AMD Zen+)

---
Built with ❤️ and Rust. AI-powered optimization for the real world.
