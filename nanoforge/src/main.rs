mod assembler;
mod hot_function;
mod jit_memory;
mod profiler;
mod safety;
mod sandbox;

use assembler::CodeGenerator;
use hot_function::HotFunction;
use jit_memory::DualMappedMemory;
use profiler::Profiler;
use std::sync::Arc;

fn main() {
    println!("NanoForge: Phase 6 - The Profiler");

    safety::install_signal_handler();

    let page_size = 4096;

    // --- Step 1: Generate Code ---
    // We'll generate a loop to make it "hot" enough to measure
    // fn(iterations: i64) -> i64
    // loop: dec rdi; jnz loop; ret
    // This is hard to generate with just add_n.
    // Let's stick to add_n but call it many times.

    println!("Generating 'Add 1' variant...");
    let code_a_bytes = CodeGenerator::generate_add_n(1).unwrap();
    let mem_a = DualMappedMemory::new(page_size).unwrap();
    CodeGenerator::emit_to_memory(&mem_a, &code_a_bytes, 0);
    let hot_func = Arc::new(HotFunction::new(mem_a, 0));

    // --- Step 2: Profile It ---
    // Note: perf_event_open usually requires CAP_PERFMON or root, or paranoid level <= 1.
    // If this fails, we'll print a warning.

    let profiler = match Profiler::new_instruction_counter() {
        Ok(p) => {
            println!("Profiler initialized (Instructions).");
            Some(p)
        }
        Err(e) => {
            println!("WARNING: Profiler init failed (are you root?): {}", e);
            None
        }
    };

    if let Some(p) = &profiler {
        p.enable();
    }

    let mut total_result = 0;
    for _ in 0..1_000_000 {
        total_result += hot_func.call(1);
    }

    if let Some(p) = &profiler {
        p.disable();
        let count = p.read();
        println!("Profiled 1,000,000 calls.");
        println!("Total Instructions Executed: {}", count);
        // 1M calls * (mov + add + ret + overhead) ~= 3-5M instructions?
    }

    println!("Result: {}", total_result);
    println!("Phase 6 Complete.");
}
