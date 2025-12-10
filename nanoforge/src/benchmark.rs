use crate::compiler::Compiler;
use crate::jit_memory::DualMappedMemory;
use crate::parser::Parser;
use std::hint::black_box;
use std::mem;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::_rdtsc;

pub fn run_benchmark(script: &str, iterations: usize, opt_level: u8) -> Result<(), String> {
    println!("Benchmarking script ({} iterations)...", iterations);

    // 1. Parse
    let mut parser = Parser::new();
    let program = parser
        .parse(script)
        .map_err(|e| format!("Parse error: {}", e))?;

    // 2. Compile
    let (code, start_offset) = Compiler::compile_program(&program, opt_level)?;

    // 3. JIT Memory
    let memory =
        DualMappedMemory::new(code.len() + 4096).map_err(|e| format!("Memory error: {}", e))?;

    // Emit code
    crate::assembler::CodeGenerator::emit_to_memory(&memory, &code, 0);

    // 4. Get Function Pointer
    let func_ptr = unsafe { memory.rx_ptr.add(start_offset) };
    let func: extern "C" fn() -> i64 = unsafe { mem::transmute(func_ptr) };

    println!("Code compiled. Size: {} bytes. executing...", code.len());

    // 5. Warmup
    println!("Warming up...");
    for _ in 0..100 {
        black_box(func());
    }

    // 6. Benchmark
    println!("Running benchmark loop...");

    let start_cycles = unsafe { _rdtsc() };

    for _ in 0..iterations {
        black_box(func());
    }

    let end_cycles = unsafe { _rdtsc() };

    let total_cycles = end_cycles - start_cycles;
    let avg_cycles = total_cycles as f64 / iterations as f64;

    println!("---------------------------------------------------");
    println!("Total Cycles: {}", total_cycles);
    println!("Iterations:   {}", iterations);
    println!("Avg Cycles/Op: {:.2}", avg_cycles);
    println!("---------------------------------------------------");

    Ok(())
}
