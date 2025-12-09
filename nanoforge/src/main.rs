use nanoforge::assembler::CodeGenerator;
use nanoforge::hot_function::HotFunction;
use nanoforge::jit_memory::DualMappedMemory;
use nanoforge::optimizer::Optimizer;
use nanoforge::profiler::Profiler;
use nanoforge::safety;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() {
    println!("NanoForge: Phase 8 - Heuristic Engine");

    safety::install_signal_handler();

    let page_size = 4096;

    // --- Step 1: Initial State (Simple Loop) ---
    println!("Initializing with 'Simple Loop' variant...");
    // We calculate sum(0..1000)
    let code_a_bytes = CodeGenerator::generate_sum_loop().unwrap();
    let mem_a = DualMappedMemory::new(page_size).unwrap();
    CodeGenerator::emit_to_memory(&mem_a, &code_a_bytes, 0);
    let hot_func = Arc::new(HotFunction::new(mem_a, 0));

    // --- Step 2: Initialize Profiler ---
    // Try to connect to daemon first
    let pid = std::process::id() as i32;
    let profiler: Arc<dyn nanoforge::profiler::ProfileSource> =
        match nanoforge::profiler::RemoteProfiler::new(pid) {
            Ok(p) => {
                println!("Connected to NanoForge Daemon.");
                Arc::new(p)
            }
            Err(e) => {
                println!(
                    "Daemon connection failed ({}), falling back to local profiler.",
                    e
                );
                match Profiler::new_instruction_counter(0) {
                    Ok(p) => {
                        println!("Local Profiler initialized.");
                        p.enable();
                        Arc::new(p)
                    }
                    Err(e) => {
                        println!("WARNING: Profiler init failed: {}", e);
                        return;
                    }
                }
            }
        };

    // --- Step 3: Start Optimizer ---
    // Trigger optimization based on internal heuristics
    let optimizer = Optimizer::new(hot_func.clone(), profiler.clone());
    optimizer.start_background_thread();

    // --- Step 4: Workload ---
    println!("Starting workload (Summing 0..1000 repeatedly)...");
    let mut total_result = 0;

    // Run enough to hit > 50M instructions
    // Each call is ~1000 instructions (roughly).
    // We need > 50,000 calls * 1000 = 50M.
    // Let's do 100 steps of 10,000 calls = 1M per step.
    // Total = 100M instructions.
    for i in 0..100 {
        let mut batch_sum = 0;
        // 10,000 calls * 1000 iterations
        for _ in 0..10000 {
            // Ensure input is multiple of 8 for our simple AVX2 impl
            batch_sum += hot_func.call(1000);
        }
        total_result += batch_sum;

        let current_count = profiler.read();
        println!(
            "Step {}: Instructions = {}, Last Batch Sum = {}",
            i, current_count, batch_sum
        );

        thread::sleep(Duration::from_millis(100));
    }

    profiler.disable();
    println!("Final Result: {}", total_result);
    println!("Phase 10 Complete.");
}
