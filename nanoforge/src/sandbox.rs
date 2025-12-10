//! Nanosecond Sandbox Benchmarker
//!
//! Provides cycle-accurate benchmarking for JIT-compiled code variants.
//! Uses perf_event counters and RDTSC for precise measurements.

#![allow(dead_code)]
use crate::profiler::Profiler;
use crate::variant_generator::CompiledVariant;
use std::hint::black_box;
use std::mem;
use std::time::Instant;

/// Result of benchmarking a single variant
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub cycles_per_op: u64,
    pub nanoseconds_per_op: u64,
    pub instructions: u64,
    pub iterations: u64,
}

impl BenchmarkResult {
    pub fn throughput_ops_per_sec(&self) -> f64 {
        if self.nanoseconds_per_op == 0 {
            return 0.0;
        }
        1_000_000_000.0 / self.nanoseconds_per_op as f64
    }
}

/// A ranked variant with benchmark results
#[derive(Debug)]
pub struct RankedVariant {
    pub rank: usize,
    pub variant_name: String,
    pub result: BenchmarkResult,
}

/// Configuration for the nanosecond sandbox
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub warmup_iterations: u32,
    pub measurement_iterations: u32,
    pub pin_to_core: Option<usize>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measurement_iterations: 1000,
            pin_to_core: Some(0),
        }
    }
}

/// Nanosecond-precision sandbox for benchmarking code variants
pub struct NanosecondSandbox {
    config: SandboxConfig,
}

impl NanosecondSandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    /// Pin the current thread to a specific CPU core for consistent measurements
    pub fn pin_thread(&self) -> Result<(), String> {
        if let Some(core_id) = self.config.pin_to_core {
            pin_thread_to_core(core_id)?;
        }
        Ok(())
    }

    /// Benchmark a compiled variant with the given input
    pub fn benchmark(&self, variant: &CompiledVariant, input: u64) -> BenchmarkResult {
        // Pin thread for consistent results
        let _ = self.pin_thread();

        // Warmup phase - fill caches, stabilize branch predictors
        for _ in 0..self.config.warmup_iterations {
            black_box(variant.execute(input));
        }

        // Memory fence before measurement
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // Measure with RDTSC
        let start_cycles = rdtsc();
        let start_time = Instant::now();

        for _ in 0..self.config.measurement_iterations {
            black_box(variant.execute(input));
        }

        let end_cycles = rdtsc();
        let elapsed = start_time.elapsed();

        let total_cycles = end_cycles.saturating_sub(start_cycles);
        let iterations = self.config.measurement_iterations as u64;

        BenchmarkResult {
            cycles_per_op: total_cycles / iterations,
            nanoseconds_per_op: elapsed.as_nanos() as u64 / iterations,
            instructions: 0, // Would need perf counter
            iterations,
        }
    }

    /// Benchmark with perf counters for detailed metrics
    pub fn benchmark_with_perf(
        &self,
        variant: &CompiledVariant,
        input: u64,
    ) -> Result<BenchmarkResult, String> {
        // Pin thread
        let _ = self.pin_thread();

        // Try to create profiler (may fail without CAP_PERFMON)
        let profiler = Profiler::new_instruction_counter(0)?;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            black_box(variant.execute(input));
        }

        // Measurement with perf
        profiler.enable();
        let start_cycles = rdtsc();
        let start_time = Instant::now();

        for _ in 0..self.config.measurement_iterations {
            black_box(variant.execute(input));
        }

        let end_cycles = rdtsc();
        let elapsed = start_time.elapsed();
        profiler.disable();

        let instructions = profiler.read();
        let iterations = self.config.measurement_iterations as u64;

        Ok(BenchmarkResult {
            cycles_per_op: (end_cycles.saturating_sub(start_cycles)) / iterations,
            nanoseconds_per_op: elapsed.as_nanos() as u64 / iterations,
            instructions: instructions / iterations,
            iterations,
        })
    }

    /// Benchmark all variants and return ranked results
    pub fn benchmark_all(&self, variants: &[CompiledVariant], input: u64) -> Vec<RankedVariant> {
        let mut results: Vec<_> = variants
            .iter()
            .map(|v| {
                let result = self.benchmark(v, input);
                (v.config.name.clone(), result)
            })
            .collect();

        // Sort by cycles per op (lower is better)
        results.sort_by_key(|(_, r)| r.cycles_per_op);

        results
            .into_iter()
            .enumerate()
            .map(|(rank, (name, result))| RankedVariant {
                rank,
                variant_name: name,
                result,
            })
            .collect()
    }

    /// Find the fastest variant
    pub fn find_fastest<'a>(
        &self,
        variants: &'a [CompiledVariant],
        input: u64,
    ) -> Option<(&'a CompiledVariant, BenchmarkResult)> {
        if variants.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_result = self.benchmark(&variants[0], input);

        for (idx, variant) in variants.iter().enumerate().skip(1) {
            let result = self.benchmark(variant, input);
            if result.cycles_per_op < best_result.cycles_per_op {
                best_idx = idx;
                best_result = result;
            }
        }

        Some((&variants[best_idx], best_result))
    }
}

impl Default for NanosecondSandbox {
    fn default() -> Self {
        Self::new(SandboxConfig::default())
    }
}

/// Read the Time Stamp Counter (TSC) for cycle-accurate timing
#[inline(always)]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let lo: u32;
        let hi: u32;
        std::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem)
        );
        ((hi as u64) << 32) | (lo as u64)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64
        std::time::Instant::now().elapsed().as_nanos() as u64
    }
}

/// Pin the current thread to a specific CPU core
pub fn pin_thread_to_core(core_id: usize) -> Result<(), String> {
    unsafe {
        let mut cpuset: libc::cpu_set_t = mem::zeroed();
        libc::CPU_SET(core_id, &mut cpuset);

        let pid = 0; // 0 means current thread
        let ret = libc::sched_setaffinity(pid, mem::size_of::<libc::cpu_set_t>(), &cpuset);

        if ret != 0 {
            return Err(format!("Failed to pin thread to core {}", core_id));
        }
    }
    Ok(())
}

/// Simple benchmark without variant infrastructure
pub fn benchmark_function(func: extern "C" fn(i64) -> i64, input: i64, iterations: u64) -> u128 {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(func(input));
    }
    start.elapsed().as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc() {
        let t1 = rdtsc();
        // Do some work
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        black_box(sum);
        let t2 = rdtsc();

        assert!(t2 > t1, "RDTSC should increase monotonically");
        println!("RDTSC delta: {} cycles", t2 - t1);
    }

    #[test]
    fn test_pin_thread() {
        // This may fail without permissions, which is OK
        let result = pin_thread_to_core(0);
        println!("Pin thread result: {:?}", result);
    }
}
