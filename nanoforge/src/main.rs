mod assembler;
mod hot_function;
mod jit_memory;
mod optimizer;
mod profiler;
mod safety;
mod sandbox;

use assembler::CodeGenerator;
use hot_function::HotFunction;
use jit_memory::DualMappedMemory;
use optimizer::Optimizer;
use profiler::Profiler;
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
    let profiler = match Profiler::new_instruction_counter() {
        Ok(p) => {
            println!("Profiler initialized.");
            Arc::new(p)
        }
        Err(e) => {
            println!("WARNING: Profiler init failed: {}", e);
            return;
        }
    };

    profiler.enable();

    // --- Step 3: Start Optimizer ---
    // Trigger optimization after 10M instructions (loops generate more instructions)
    let optimizer = Optimizer::new(hot_func.clone(), profiler.clone(), 10_000_000);
    optimizer.start_background_thread();

    // --- Step 4: Workload ---
    println!("Starting workload (Summing 0..1000 repeatedly)...");
    let mut total_result = 0;

    // 20 steps
    for i in 0..20 {
        let mut batch_sum = 0;
        // Generate load.
        // Each call does a loop of 1000 iterations.
        // 1000 iterations * ~4 instructions = 4000 instructions per call.
        // 5000 calls = 20M instructions.
        for _ in 0..5000 {
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
    println!("Phase 9 Complete.");
}
