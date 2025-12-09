use clap::Parser;
use nanoforge::assembler::CodeGenerator;
use nanoforge::hot_function::HotFunction;
use nanoforge::jit_memory::DualMappedMemory;
use nanoforge::optimizer::Optimizer;
use nanoforge::profiler::Profiler;
use nanoforge::safety;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Unix Domain Socket
    #[arg(short, long, default_value = "/tmp/nanoforge.sock")]
    socket_path: String,

    /// Threshold for Unrolled Loop optimization
    #[arg(long, default_value_t = 10_000_000)]
    threshold_unrolled: u64,

    /// Threshold for AVX2 optimization
    #[arg(long, default_value_t = 50_000_000)]
    threshold_avx2: u64,
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Register Crash Handler (P0 Safety)
    nanoforge::safety::register_crash_handler();

    let args = Args::parse();

    info!("NanoForge: Phase 8 - Heuristic Engine");
    info!(
        "Configuration: Socket={}, Unrolled={}, AVX2={}",
        args.socket_path, args.threshold_unrolled, args.threshold_avx2
    );

    let page_size = 4096;

    // --- Step 1: Initial State (Simple Loop) ---
    info!("Initializing with 'Simple Loop' variant...");
    // We calculate sum(0..1000)
    // We calculate sum(0..1000)
    let code_a_bytes = CodeGenerator::generate_sum_loop().expect("Failed to generate initial code");
    let mem_a = DualMappedMemory::new(page_size).expect("Failed to allocate JIT memory");
    CodeGenerator::emit_to_memory(&mem_a, &code_a_bytes, 0);
    let hot_func = Arc::new(HotFunction::new(mem_a, 0));

    // --- Step 2: Initialize Profiler ---
    // Try to connect to daemon first
    let pid = std::process::id() as i32;
    let profiler: Arc<dyn nanoforge::profiler::ProfileSource> =
        match nanoforge::profiler::RemoteProfiler::new(pid) {
            Ok(p) => {
                info!("Connected to NanoForge Daemon.");
                Arc::new(p)
            }
            Err(e) => {
                warn!(
                    "Daemon connection failed ({}), falling back to local profiler.",
                    e
                );
                match Profiler::new_instruction_counter(0) {
                    Ok(p) => {
                        info!("Local Profiler initialized.");
                        p.enable();
                        Arc::new(p)
                    }
                    Err(e) => {
                        error!("Profiler init failed: {}", e);
                        return;
                    }
                }
            }
        };

    // --- Step 3: Start Optimizer ---
    // Trigger optimization based on internal heuristics
    let optimizer = Optimizer::new(
        hot_func.clone(),
        profiler.clone(),
        args.threshold_unrolled,
        args.threshold_avx2,
    );
    optimizer.start_background_thread();

    // --- Step 4: Workload ---
    info!("Starting workload (Summing 0..1000 repeatedly)...");
    let mut total_result = 0;

    // Run enough to hit > 50M instructions
    for i in 0..100 {
        let mut batch_sum = 0;
        // 10,000 calls * 1000 iterations
        for _ in 0..10000 {
            // Ensure input is multiple of 8 for our simple AVX2 impl
            batch_sum += hot_func.call(1000);
        }
        total_result += batch_sum;

        let current_count = profiler.read();
        info!(
            "Step {}: Instructions = {}, Last Batch Sum = {}",
            i, current_count, batch_sum
        );

        thread::sleep(Duration::from_millis(100));
    }

    profiler.disable();
    info!("Final Result: {}", total_result);
    info!("Phase 10 Complete.");
}
