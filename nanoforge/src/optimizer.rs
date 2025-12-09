use crate::assembler::CodeGenerator;
use crate::hot_function::HotFunction;
use crate::jit_memory::DualMappedMemory;
use crate::profiler::Profiler;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub struct Optimizer {
    hot_function: Arc<HotFunction>,
    profiler: Arc<Profiler>,
    threshold: u64,
}

impl Optimizer {
    pub fn new(hot_function: Arc<HotFunction>, profiler: Arc<Profiler>, threshold: u64) -> Self {
        Optimizer {
            hot_function,
            profiler,
            threshold,
        }
    }

    pub fn start_background_thread(self) {
        thread::spawn(move || {
            println!("Optimizer: Background thread started.");
            let mut optimized = false;

            loop {
                thread::sleep(Duration::from_millis(100));

                if optimized {
                    continue;
                }

                let count = self.profiler.read();
                if count > self.threshold {
                    println!(
                        "Optimizer: Threshold reached ({} > {}). Optimizing...",
                        count, self.threshold
                    );

                    // Heuristic: If hot, switch to 'Unrolled Loop' variant.
                    match CodeGenerator::generate_sum_loop_unrolled() {
                        Ok(code) => match DualMappedMemory::new(4096) {
                            Ok(mem) => {
                                CodeGenerator::emit_to_memory(&mem, &code, 0);
                                self.hot_function.update(mem, 0);
                                println!("Optimizer: Optimization applied (Unrolled Loop).");
                                optimized = true;
                            }
                            Err(e) => println!("Optimizer Error: Memory allocation failed: {}", e),
                        },
                        Err(e) => println!("Optimizer Error: Code gen failed: {}", e),
                    }
                }
            }
        });
    }
}
