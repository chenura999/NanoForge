use crate::assembler::CodeGenerator;
use crate::hot_function::HotFunction;
use crate::jit_memory::DualMappedMemory;
use crate::profiler::ProfileSource; // Changed import
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
enum OptimizationLevel {
    Baseline,
    Unrolled,
    Avx2,
}

pub struct Optimizer {
    hot_function: Arc<HotFunction>,
    profiler: Arc<dyn ProfileSource>, // Changed type
                                      // We'll use hardcoded thresholds for this demo, or we could pass them in.
                                      // threshold: u64, // Removed single threshold
}

impl Optimizer {
    pub fn new(hot_function: Arc<HotFunction>, profiler: Arc<dyn ProfileSource>) -> Self {
        Optimizer {
            hot_function,
            profiler,
        }
    }

    pub fn start_background_thread(self) {
        thread::spawn(move || {
            println!("Optimizer: Background thread started.");
            let mut current_level = OptimizationLevel::Baseline;

            // Thresholds
            const THRESHOLD_UNROLLED: u64 = 10_000_000;
            const THRESHOLD_AVX2: u64 = 50_000_000;

            loop {
                thread::sleep(Duration::from_millis(100));

                let count = self.profiler.read();

                match current_level {
                    OptimizationLevel::Baseline => {
                        if count > THRESHOLD_UNROLLED {
                            println!(
                                "Optimizer: Threshold 1 reached ({} > {}). Upgrading to Unrolled Loop...",
                                count, THRESHOLD_UNROLLED
                            );
                            match CodeGenerator::generate_sum_loop_unrolled() {
                                Ok(code) => self.apply_optimization(
                                    code,
                                    OptimizationLevel::Unrolled,
                                    &mut current_level,
                                ),
                                Err(e) => println!("Optimizer Error: Code gen failed: {}", e),
                            }
                        }
                    }
                    OptimizationLevel::Unrolled => {
                        if count > THRESHOLD_AVX2 {
                            println!(
                                "Optimizer: Threshold 2 reached ({} > {}). Upgrading to AVX2...",
                                count, THRESHOLD_AVX2
                            );

                            if is_x86_feature_detected!("avx2") {
                                match CodeGenerator::generate_sum_avx2() {
                                    Ok(code) => self.apply_optimization(
                                        code,
                                        OptimizationLevel::Avx2,
                                        &mut current_level,
                                    ),
                                    Err(e) => println!("Optimizer Error: Code gen failed: {}", e),
                                }
                            } else {
                                println!(
                                    "Optimizer: AVX2 not supported. Staying at Unrolled level."
                                );
                                // Mark as AVX2 so we don't keep trying, effectively "maxed out"
                                current_level = OptimizationLevel::Avx2;
                            }
                        }
                    }
                    OptimizationLevel::Avx2 => {
                        // Already at max optimization
                        continue;
                    }
                }
            }
        });
    }

    fn apply_optimization(
        &self,
        code: Vec<u8>,
        new_level: OptimizationLevel,
        current_level: &mut OptimizationLevel,
    ) {
        match DualMappedMemory::new(4096) {
            Ok(mem) => {
                CodeGenerator::emit_to_memory(&mem, &code, 0);
                self.hot_function.update(mem, 0);
                println!("Optimizer: Optimization applied ({:?}).", new_level);
                *current_level = new_level;
            }
            Err(e) => println!("Optimizer Error: Memory allocation failed: {}", e),
        }
    }
}
