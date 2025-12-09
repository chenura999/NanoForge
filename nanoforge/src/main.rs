mod hot_function;
mod jit_memory;
mod safety;
mod sandbox;

use jit_memory::DualMappedMemory;
use std::ptr;

fn main() {
    println!("NanoForge: Phase 4 - Stability & Safety Rails");

    // 1. Install Signal Handler
    safety::install_signal_handler();
    println!("Signal handler installed.");

    let page_size = 4096;
    let mem = DualMappedMemory::new(page_size).unwrap();

    // 2. Generate "Bad" Code (SIGILL)
    // 0F 0B is UD2 (Undefined Instruction) on x86
    let bad_code: [u8; 2] = [0x0f, 0x0b];

    unsafe {
        ptr::copy_nonoverlapping(bad_code.as_ptr(), mem.rw_ptr, bad_code.len());
    }
    mem.flush_icache();
    println!("Wrote 'Bad' Code (UD2 instruction).");

    // 3. Execute Safely
    let func: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(mem.rx_ptr) };

    println!("Attempting to execute bad code inside sandbox...");
    let result = safety::run_safely(|| {
        println!("  -> Inside sandbox, calling function...");
        func(0)
    });

    match result {
        Ok(val) => println!("Success: {}", val),
        Err(e) => println!("SUCCESSFULLY CAUGHT CRASH: {}", e),
    }

    println!("Main thread continues... System survived.");
}
