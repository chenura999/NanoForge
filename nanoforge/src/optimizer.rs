use crate::assembler::CodeGenerator;
use crate::benchmarker::Benchmarker;
use crate::hot_function::HotFunction;
use crate::jit_memory::DualMappedMemory;
use crate::profiler::ProfileSource;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq)]
enum OptimizationLevel {
    Baseline,
    Optimized,
}

struct Candidate {
    name: String,
    code: Vec<u8>,
}

pub struct Optimizer {
    hot_function: Arc<HotFunction>,
    profiler: Arc<dyn ProfileSource>,
    optimization_trigger: u64,
}

impl Optimizer {
    pub fn new(
        hot_function: Arc<HotFunction>,
        profiler: Arc<dyn ProfileSource>,
        optimization_trigger: u64,
        _unused_avx2_threshold: u64, // Keep signature compatible for now
    ) -> Self {
        Optimizer {
            hot_function,
            profiler,
            optimization_trigger,
        }
    }

    pub fn start_background_thread(self) {
        thread::spawn(move || {
            info!("Optimizer: Background thread started.");
            let mut current_level = OptimizationLevel::Baseline;

            loop {
                thread::sleep(Duration::from_millis(100));

                let count = self.profiler.read();

                if current_level == OptimizationLevel::Baseline && count > self.optimization_trigger
                {
                    info!(
                        "Optimizer: Trigger reached ({} > {}). Starting Benchmark Sandbox...",
                        count, self.optimization_trigger
                    );

                    // 1. Generate Candidates
                    let mut candidates = Vec::new();

                    // Candidate A: Unrolled Loop
                    if let Ok(code) = CodeGenerator::generate_sum_loop_unrolled() {
                        candidates.push(Candidate {
                            name: "Unrolled Loop".to_string(),
                            code,
                        });
                    }

                    // Candidate B: AVX2 (if supported)
                    if is_x86_feature_detected!("avx2") {
                        if let Ok(code) = CodeGenerator::generate_sum_avx2() {
                            candidates.push(Candidate {
                                name: "AVX2 SIMD".to_string(),
                                code,
                            });
                        }
                    }

                    if candidates.is_empty() {
                        info!("Optimizer: No candidates available.");
                        current_level = OptimizationLevel::Optimized; // Stop trying
                        continue;
                    }

                    // 2. Race Candidates
                    let mut best_candidate: Option<&Candidate> = None;
                    let mut best_score = u64::MAX;

                    let page_size = 4096;
                    // We need a temporary memory to run benchmarks
                    let sandbox_mem =
                        DualMappedMemory::new(page_size).expect("Sandbox alloc failed");

                    for candidate in &candidates {
                        // Emit candidate code to sandbox
                        CodeGenerator::emit_to_memory(&sandbox_mem, &candidate.code, 0);

                        // Get function pointer
                        let func: extern "C" fn(u64) -> u64 =
                            unsafe { std::mem::transmute(sandbox_mem.rx_ptr) };

                        // Measure
                        // Input 1000, 1000 iterations
                        let cycles = unsafe { Benchmarker::measure(func, 1000, 1000) };

                        info!(
                            "Optimizer: Candidate '{}' -> {} cycles/iter",
                            candidate.name, cycles
                        );

                        if cycles < best_score {
                            best_score = cycles;
                            best_candidate = Some(candidate);
                        }
                    }

                    // 3. Pick Winner
                    if let Some(winner) = best_candidate {
                        info!(
                            "Optimizer: Winner is '{}' with {} cycles.",
                            winner.name, best_score
                        );
                        self.apply_optimization(&winner.code);
                        current_level = OptimizationLevel::Optimized;
                    }
                }
            }
        });
    }

    fn apply_optimization(&self, code: &[u8]) {
        // Allocate new executable memory for the hot path
        let page_size = 4096;
        let new_mem = DualMappedMemory::new(page_size).unwrap();

        // Emit the optimized code
        CodeGenerator::emit_to_memory(&new_mem, code, 0);

        // Hot-swap the function pointer!
        // The old memory will be dropped when the Arc count goes to zero (eventually)
        // Note: In a real system, we need RCU or safe reclamation.
        // Here we just swap. The `HotFunction` struct holds the `current_mem`.
        self.hot_function.update(new_mem, 0);

        info!("Optimizer: Optimization applied.");
    }
}
