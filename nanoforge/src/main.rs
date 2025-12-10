use clap::{Parser, Subcommand};
use nanoforge::assembler::CodeGenerator;
use nanoforge::compiler::Compiler;
use nanoforge::hot_function::HotFunction;
use nanoforge::jit_memory::DualMappedMemory;
use nanoforge::optimizer::Optimizer;
use nanoforge::parser::Parser as NanoParser;
use nanoforge::profiler::Profiler;
use std::io::{self, Write};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

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

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the interactive REPL
    Repl,
    /// Run a script file
    Run { file: String },
    /// Run the internal demo/benchmark
    Demo,
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Register Crash Handler
    nanoforge::safety::register_crash_handler();

    let args = Args::parse();

    match &args.command {
        Some(Commands::Repl) => run_repl(),
        Some(Commands::Run { file }) => run_file(file),
        Some(Commands::Demo) => run_demo(&args),
        None => run_repl(), // Default to REPL if no args
    }
}

fn run_repl() {
    println!("NanoForge REPL v0.1.0");
    println!("Type 'RUN' to execute buffer, 'CLEAR' to reset, 'EXIT' to quit.");

    let mut buffer = String::new();
    let stdin = io::stdin();

    loop {
        print!(">> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            break;
        }

        let trimmed = line.trim();
        match trimmed {
            "EXIT" => break,
            "CLEAR" => {
                buffer.clear();
                println!("Buffer cleared.");
            }
            "RUN" => {
                println!("Compiling...");
                execute_script(&buffer);
                buffer.clear();
            }
            _ => {
                buffer.push_str(&line);
            }
        }
    }
}

fn run_file(path: &str) {
    let content = std::fs::read_to_string(path).expect("Failed to read file");
    match execute_script(&content) {
        Ok(_) => {}
        Err(e) => println!("Error: {}", e),
    }
}

fn execute_script(script: &str) -> Result<(), String> {
    let mut parser = NanoParser::new();
    match parser.parse(script) {
        Ok(prog) => {
            let (code, main_offset) =
                Compiler::compile_program(&prog).map_err(|e| format!("{}", e))?;

            // Debug Dump
            std::fs::write("debug.bin", &code).expect("Failed to write debug.bin");
            println!("Dumped machine code to debug.bin");

            let memory = DualMappedMemory::new(code.len() + 4096).map_err(|e| format!("{}", e))?;
            CodeGenerator::emit_to_memory(&memory, &code, 0);
            let func_ptr: extern "C" fn() -> i64 =
                unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
            println!("Executing...");
            let result = func_ptr();
            println!("Result: {}", result);
            Ok(())
        }
        Err(e) => Err(format!("Parsing Error: {}", e)),
    }
}

fn run_demo(args: &Args) {
    // Initialize Metrics (Prometheus) - Only needed for long running demo
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(([0, 0, 0, 0], 9000))
        .install()
        .ok(); // Ignore if already installed

    info!("NanoForge: Phase 8 - Heuristic Engine");
    info!(
        "Configuration: Socket={}, Unrolled={}, AVX2={}",
        args.socket_path, args.threshold_unrolled, args.threshold_avx2
    );

    let page_size = 4096;

    // --- Step 1: Initial State (Simple Loop) ---
    info!("Initializing with 'Simple Loop' variant...");
    let code_a_bytes = CodeGenerator::generate_sum_loop().expect("Failed to generate initial code");
    let mem_a = DualMappedMemory::new(page_size).expect("Failed to allocate JIT memory");
    CodeGenerator::emit_to_memory(&mem_a, &code_a_bytes, 0);
    let hot_func = Arc::new(HotFunction::new(mem_a, 0));

    // --- Step 2: Initialize Profiler ---
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

    for i in 0..100 {
        let mut batch_sum = 0;
        for _ in 0..10000 {
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
