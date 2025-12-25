use clap::{Parser, Subcommand};
use nanoforge::ai_optimizer::{ContextualBandit, OptimizationFeatures, SizeBucket, VariantBandit};
use nanoforge::assembler::CodeGenerator;
use nanoforge::compiler::Compiler;
use nanoforge::cpu_features::CpuFeatures;
use nanoforge::hot_function::HotFunction;
use nanoforge::jit_memory::DualMappedMemory;
use nanoforge::sandbox::{NanosecondSandbox, SandboxConfig};
use nanoforge::variant_generator::VariantGenerator;

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
    Run {
        file: String,
        #[arg(short, long, default_value_t = 3)]
        level: u8,
    },
    /// Run the internal demo/benchmark
    Demo,
    /// Benchmark a script file (10k iterations)
    Benchmark {
        file: String,
        #[arg(short, long, default_value_t = 3)]
        level: u8,
    },
    /// Run Adaptive Optimization Demo
    Adaptive { file: String },
    /// Run SOAE (Self-Optimizing Assembly Engine) Demo
    Soae { file: String },
    /// Run SOAE with AI-Powered Variant Selection
    SoaeAi {
        file: String,
        /// Number of learning iterations
        #[arg(short, long, default_value_t = 50)]
        iterations: u32,
    },
    /// Run SOAE with Contextual Bandit (learns decision boundaries)
    SoaeContext {
        file: String,
        /// Number of learning iterations
        #[arg(short, long, default_value_t = 100)]
        iterations: u32,
    },
    /// 🧬 EVOLVE: Use genetic algorithms to evolve optimal code
    Evolve {
        file: String,
        /// Number of generations to evolve
        #[arg(short, long, default_value_t = 50)]
        generations: u32,
        /// Population size
        #[arg(short, long, default_value_t = 30)]
        population: usize,
        /// Target speedup to achieve (stops early if reached)
        #[arg(short, long)]
        target: Option<f64>,
    },
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Register Crash Handler
    nanoforge::safety::register_crash_handler();

    let args = Args::parse();

    match &args.command {
        Some(Commands::Repl) => run_repl(),
        Some(Commands::Run { file, level }) => run_file(file, *level),
        Some(Commands::Demo) => run_demo(&args),
        Some(Commands::Benchmark { file, level }) => {
            let script = std::fs::read_to_string(file).expect("Failed to read file");
            // Default level 2 for explicit benchmark
            if let Err(e) = nanoforge::benchmark::run_benchmark(&script, 10_000, *level) {
                println!("Benchmark Error: {}", e);
            }
        }
        Some(Commands::Adaptive { file }) => run_adaptive(file),
        Some(Commands::Soae { file }) => run_soae(file),
        Some(Commands::SoaeAi { file, iterations }) => run_soae_ai(file, *iterations),
        Some(Commands::SoaeContext { file, iterations }) => run_soae_context(file, *iterations),
        Some(Commands::Evolve {
            file,
            generations,
            population,
            target,
        }) => run_evolve(file, *generations, *population, *target),
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
                execute_script(&buffer, 3).unwrap_or_else(|e| println!("Execution Error: {}", e));
                buffer.clear();
            }
            _ => {
                buffer.push_str(&line);
            }
        }
    }
}

fn run_file(path: &str, level: u8) {
    let content = std::fs::read_to_string(path).expect("Failed to read file");
    match execute_script(&content, level) {
        Ok(_) => {}
        Err(e) => println!("Error: {}", e),
    }
}

fn execute_script(script: &str, level: u8) -> Result<(), String> {
    let mut parser = NanoParser::new();
    match parser.parse(script) {
        Ok(prog) => {
            let (code, main_offset) =
                Compiler::compile_program(&prog, level).map_err(|e| e.to_string())?;

            // Debug Dump
            std::fs::write("debug.bin", &code).expect("Failed to write debug.bin");
            println!("Dumped machine code to debug.bin");

            let memory = DualMappedMemory::new(code.len() + 4096).map_err(|e| e.to_string())?;
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

fn run_adaptive(path: &str) {
    println!("=== NanoForge Adaptive Runtime ===");
    let script = std::fs::read_to_string(path).expect("Failed to read file");
    let mut parser = NanoParser::new();
    let prog_ir = parser.parse(&script).expect("Parse failed");

    // Constants for Metric Calculation
    // Assuming vec_add_stress.nf: 100 * 10,000 = 1,000,000 Ops per Call
    const OPS_PER_CALL: f64 = 1_000_000.0;
    const CLOCK_SPEED: f64 = 4_000_000_000.0; // 4.0 GHz reference

    // Phase 1: Tier 1 (Scalar / Level 2)
    print!("Running Tier 1 (Scalar)... ");
    io::stdout().flush().unwrap();

    let (code_base, main_offset_base) =
        Compiler::compile_program(&prog_ir, 2).expect("Compile failed");
    let mem_base = DualMappedMemory::new(code_base.len() + 4096).unwrap();
    CodeGenerator::emit_to_memory(&mem_base, &code_base, 0);

    let current_fn: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(mem_base.rx_ptr.add(main_offset_base)) };

    // Warmup
    for _ in 0..10 {
        std::hint::black_box(current_fn());
    }

    // Measure Tier 1
    let iterations = 100;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(current_fn());
    }
    let dur_t1 = start.elapsed();
    let cyc_op_t1 = (dur_t1.as_secs_f64() * CLOCK_SPEED) / (iterations as f64 * OPS_PER_CALL);

    println!("{:.2} cycles/op", cyc_op_t1);

    // Phase 2: Optimization Trigger
    println!("\n🔥 HOT SWAP TRIGGERED 🔥\n");

    // Compile Tier 2 (Vector / Level 3)
    print!("Running Tier 2 (AVX2)... ");
    io::stdout().flush().unwrap();

    let (code_opt, main_offset_opt) =
        Compiler::compile_program(&prog_ir, 3).expect("Compile failed");
    let mem_opt = DualMappedMemory::new(code_opt.len() + 4096).unwrap();
    CodeGenerator::emit_to_memory(&mem_opt, &code_opt, 0);
    let fn_opt: extern "C" fn() -> i64 =
        unsafe { std::mem::transmute(mem_opt.rx_ptr.add(main_offset_opt)) };

    // Warmup
    for _ in 0..10 {
        std::hint::black_box(fn_opt());
    }

    // Measure Tier 2
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(fn_opt());
    }
    let dur_t2 = start.elapsed();
    let cyc_op_t2 = (dur_t2.as_secs_f64() * CLOCK_SPEED) / (iterations as f64 * OPS_PER_CALL);

    println!("{:.2} cycles/op", cyc_op_t2);

    // Final Report
    let speedup = dur_t1.as_secs_f64() / dur_t2.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);
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
    // --- Step 3: Start Optimizer ---
    // let optimizer = Optimizer::new(
    //     hot_func.clone(),
    //     profiler.clone(),
    //     args.threshold_unrolled,
    //     args.threshold_avx2,
    // );
    // optimizer.start_background_thread();

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

/// Self-Optimizing Assembly Engine (SOAE) Demo
///
/// This demonstrates the core SOAE concept:
/// 1. Generate multiple code variants (Scalar, AVX2 with different unroll factors)
/// 2. Benchmark all variants in the nanosecond sandbox
/// 3. Select the fastest variant
/// 4. Show comparative performance
fn run_soae(path: &str) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     🔥 NanoForge SOAE (Self-Optimizing Assembly Engine) 🔥    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect CPU features
    let cpu = CpuFeatures::detect();
    println!("🖥️  CPU Features: {}\n", cpu.summary());

    // Parse the source file
    let script = std::fs::read_to_string(path).expect("Failed to read file");
    let mut parser = NanoParser::new();
    let program = parser.parse(&script).expect("Parse failed");

    // Generate variants
    println!("📦 Generating Code Variants...");
    let generator = VariantGenerator::new();
    let variants = generator
        .generate_variants(&program)
        .expect("Variant generation failed");

    println!("   Generated {} variants:\n", variants.len());
    for (i, v) in variants.iter().enumerate() {
        println!(
            "   {}. {} (opt level: {}, {} bytes)",
            i + 1,
            v.config.name,
            v.config.optimization_level,
            v.code_size
        );
    }

    // Create sandbox and benchmark all variants
    println!("\n⏱️  Benchmarking in Nanosecond Sandbox...\n");
    let sandbox = NanosecondSandbox::new(SandboxConfig {
        warmup_iterations: 50,
        measurement_iterations: 500,
        pin_to_core: Some(0),
    });

    // Use a test input
    let test_input = 1000u64;

    let rankings = sandbox.benchmark_all(&variants, test_input);

    // Display results
    println!("┌────┬──────────────────────┬────────────────┬────────────────┐");
    println!("│ #  │ Variant              │ Cycles/Op      │ Throughput     │");
    println!("├────┼──────────────────────┼────────────────┼────────────────┤");

    let baseline_cycles = rankings
        .first()
        .map(|r| r.result.cycles_per_op)
        .unwrap_or(1);

    for ranked in &rankings {
        let speedup = if ranked.rank == 0 {
            "🏆 WINNER".to_string()
        } else {
            let ratio = ranked.result.cycles_per_op as f64 / baseline_cycles as f64;
            format!("{:.2}x slower", ratio)
        };

        println!(
            "│ {:2} │ {:20} │ {:>14} │ {:>14} │",
            ranked.rank + 1,
            &ranked.variant_name,
            format!("{} cyc", ranked.result.cycles_per_op),
            speedup
        );
    }
    println!("└────┴──────────────────────┴────────────────┴────────────────┘");

    // Execute the winning variant
    if let Some(winner) = rankings.first() {
        let winner_variant = variants
            .iter()
            .find(|v| v.config.name == winner.variant_name)
            .expect("Winner not found");

        println!("\n🚀 Executing winner: {}", winner.variant_name);
        let result = winner_variant.execute(test_input);
        println!("   Result: {}", result);
        println!("   Cycles/Op: {}", winner.result.cycles_per_op);
        println!(
            "   Ops/Second: {:.2e}",
            winner.result.throughput_ops_per_sec()
        );
    }

    println!("\n✅ SOAE Demo Complete!\n");
}

/// SOAE with AI-Powered Variant Selection
///
/// Demonstrates Thompson Sampling bandit learning in real-time:
/// 1. Generate variants
/// 2. Initialize bandit with uniform priors
/// 3. Each iteration: bandit selects variant → benchmark → update beliefs
/// 4. Watch as bandit learns which variant is best
fn run_soae_ai(path: &str, iterations: u32) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║   🧠 NanoForge AI-Powered SOAE with Thompson Sampling 🧠    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect CPU features
    let cpu = CpuFeatures::detect();
    println!("🖥️  CPU Features: {}", cpu.summary());
    println!("📊 Learning iterations: {}\n", iterations);

    // Parse and generate variants
    let script = std::fs::read_to_string(path).expect("Failed to read file");
    let mut parser = NanoParser::new();
    let program = parser.parse(&script).expect("Parse failed");

    let generator = VariantGenerator::new();
    let variants = generator
        .generate_variants(&program)
        .expect("Variant generation failed");

    println!("📦 Generated {} variants:", variants.len());
    let variant_names: Vec<String> = variants.iter().map(|v| v.config.name.clone()).collect();
    for name in &variant_names {
        println!("   • {}", name);
    }

    // Create sandbox
    let sandbox = NanosecondSandbox::new(SandboxConfig {
        warmup_iterations: 20,
        measurement_iterations: 100,
        pin_to_core: Some(0),
    });

    // Initialize Thompson Sampling bandit
    let mut bandit = VariantBandit::new(variant_names.clone());
    let test_input = 1000u64;

    // Pre-benchmark to find true best (for validation)
    let true_rankings = sandbox.benchmark_all(&variants, test_input);
    let true_best = true_rankings
        .first()
        .map(|r| r.variant_name.clone())
        .unwrap_or_default();
    let best_cycles = true_rankings
        .first()
        .map(|r| r.result.cycles_per_op)
        .unwrap_or(1);

    println!("\n🎯 True best variant (ground truth): {}\n", true_best);
    println!("🎰 Starting Thompson Sampling learning...\n");

    // Learning loop
    let mut correct_selections = 0u32;

    for i in 1..=iterations {
        // Bandit selects variant (exploration/exploitation)
        let selected_idx = bandit.select();
        let selected_variant = &variants[selected_idx];

        // Benchmark selected variant
        let result = sandbox.benchmark(selected_variant, test_input);

        // Update bandit with performance reward
        bandit.update_with_performance(selected_idx, result.cycles_per_op, best_cycles);

        // Track accuracy
        let is_correct = variant_names[selected_idx] == true_best;
        if is_correct {
            correct_selections += 1;
        }

        // Progress output (every 10 iterations)
        if i <= 5 || i % 10 == 0 || i == iterations {
            let best_guess = bandit.get_best();
            let accuracy = (correct_selections as f64 / i as f64) * 100.0;
            let marker = if variant_names[best_guess] == true_best {
                "✓"
            } else {
                "✗"
            };

            println!(
                "  Iter {:3}: Selected {:<12} | Best guess: {:<12} {} | Accuracy: {:.1}%",
                i, &variant_names[selected_idx], &variant_names[best_guess], marker, accuracy
            );
        }
    }

    // Final results
    println!("\n{}", "═".repeat(64));
    bandit.print_status();

    let final_best = bandit.get_best();
    let converged = variant_names[final_best] == true_best;

    if converged {
        println!("\n🎉 SUCCESS: Bandit correctly converged to {}!", true_best);
    } else {
        println!(
            "\n⚠️  Bandit converged to {} (true best: {})",
            variant_names[final_best], true_best
        );
    }

    // Execute winner
    let winner_variant = &variants[final_best];
    let result = winner_variant.execute(test_input);
    println!("   Result: {}", result);

    println!("\n✅ AI-Powered SOAE Complete!\n");
}

/// SOAE with Contextual Bandit - Learns Decision Boundaries
///
/// This is the KEY DEMO that shows context-aware learning:
/// - Runs with VARYING input sizes (tiny, small, medium, large)
/// - Learns that small inputs → Scalar is better
/// - Learns that large inputs → AVX2 is better
/// - Displays the learned decision boundary!
fn run_soae_context(path: &str, iterations: u32) {
    use rand::Rng;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  🧠 CONTEXTUAL BANDIT - Learning Decision Boundaries! 🧠   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect CPU features
    let cpu = CpuFeatures::detect();
    println!("🖥️  CPU Features: {}", cpu.summary());
    println!(
        "📊 Learning iterations: {} (with variable input sizes)\n",
        iterations
    );

    // Parse and generate variants
    let script = std::fs::read_to_string(path).expect("Failed to read file");
    let mut parser = NanoParser::new();
    let program = parser.parse(&script).expect("Parse failed");

    let generator = VariantGenerator::new();
    let variants = generator
        .generate_variants(&program)
        .expect("Variant generation failed");

    let variant_names: Vec<String> = variants.iter().map(|v| v.config.name.clone()).collect();
    println!("📦 Generated {} variants:", variants.len());
    for name in &variant_names {
        println!("   • {}", name);
    }

    // Create sandbox
    let sandbox = NanosecondSandbox::new(SandboxConfig {
        warmup_iterations: 10,
        measurement_iterations: 50,
        pin_to_core: Some(0),
    });

    // Initialize CONTEXTUAL bandit (one per size bucket!)
    let mut bandit = ContextualBandit::new(variant_names.clone());

    println!("\n🎰 Starting Contextual Learning with Variable Input Sizes...\n");
    println!("   The AI will see different input sizes and learn which");
    println!("   variant works best for each size bucket!\n");

    // Test sizes for each bucket
    let test_sizes: Vec<u64> = vec![
        10, 20, // Tiny
        50, 100, 200, // Small
        500, 1000, 2000, // Medium
        5000, 10000,  // Large
        100000, // Huge
    ];

    let mut rng = rand::thread_rng();

    // Learning loop with varying input sizes
    for i in 1..=iterations {
        // Randomly pick an input size
        let input_size = test_sizes[rng.gen_range(0..test_sizes.len())];
        let context = OptimizationFeatures::new(input_size);
        let bucket = context.size_bucket();

        // Contextual bandit selects based on bucket
        let selected_idx = bandit.select(&context);
        let selected_variant = &variants[selected_idx];

        // Benchmark this variant with this input size
        let result = sandbox.benchmark(selected_variant, input_size);

        // Find the actual best for this size (to compute reward)
        let rankings = sandbox.benchmark_all(&variants, input_size);
        let best_cycles = rankings
            .first()
            .map(|r| r.result.cycles_per_op)
            .unwrap_or(1);

        // Update bandit with performance in this context
        bandit.update_with_performance(&context, selected_idx, result.cycles_per_op, best_cycles);

        // Progress output
        if i <= 10 || i % 20 == 0 || i == iterations {
            println!(
                "  Iter {:3}: N={:6} ({:12}) → Selected {}",
                i,
                input_size,
                bucket.name(),
                &variant_names[selected_idx]
            );
        }
    }

    // Display the learned decision boundary!
    println!("\n{}", "═".repeat(64));
    bandit.print_decision_boundary();

    // Show detailed stats
    bandit.print_full_status();

    // Summary analysis
    println!("\n📋 Analysis:");
    let decisions = bandit.get_decision_boundary();
    let mut scalar_wins = 0;
    let mut avx_wins = 0;

    for (bucket, variant, _) in &decisions {
        let is_scalar = variant.starts_with("Scalar");
        if is_scalar {
            scalar_wins += 1;
            if matches!(bucket, SizeBucket::Tiny | SizeBucket::Small) {
                println!("   ✓ {} correctly prefers Scalar ({})", bucket, variant);
            }
        } else {
            avx_wins += 1;
            if matches!(
                bucket,
                SizeBucket::Medium | SizeBucket::Large | SizeBucket::Huge
            ) {
                println!("   ✓ {} correctly prefers AVX2 ({})", bucket, variant);
            }
        }
    }

    println!(
        "\n   Decision Summary: Scalar wins {} buckets, AVX2 wins {} buckets",
        scalar_wins, avx_wins
    );

    println!("\n✅ Contextual Bandit Learning Complete!\n");
}

/// 🧬 EVOLVE: Genetic Algorithm Code Evolution
///
/// This demonstrates self-evolving code:
/// 1. Parse seed function from .nf file
/// 2. Execute seed function to generate "Ground Truth" outputs
/// 3. Create population of mutated variants
/// 4. Evolve through selection, crossover, mutation
/// 5. Watch code get faster while maintaining correctness!
fn run_evolve(path: &str, generations: u32, population_size: usize, target: Option<f64>) {
    use nanoforge::evolution::{EvolutionConfig, EvolutionEngine};
    use nanoforge::validator::TestCase;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     🧬 NanoForge Self-Evolving JIT (Genetic Algorithm) 🧬    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse the seed function
    let script = std::fs::read_to_string(path).expect("Failed to read file");
    let mut parser = NanoParser::new();
    let program = parser.parse(&script).expect("Parse failed");

    if program.functions.is_empty() {
        println!("❌ No functions found in {}", path);
        return;
    }

    let seed_function = &program.functions[0];
    println!("🌱 Seed function: {}", seed_function.name);
    println!("   {} instructions", seed_function.instructions.len());
    for (i, instr) in seed_function.instructions.iter().enumerate() {
        println!("   {}: {:?}", i, instr);
    }
    println!("   {} arguments\n", seed_function.args.len());

    // --- Generate Ground Truth ---
    println!("🧪 Generating Ground Truth from Seed Code...");

    // Compile seed to run it
    let (code, main_offset) =
        Compiler::compile_program(&program, 0).expect("Failed to compile seed for ground truth");

    let memory = DualMappedMemory::new(code.len() + 4096).expect("Memory alloc failed");
    CodeGenerator::emit_to_memory(&memory, &code, 0);

    // Cast to function pointer
    let func_ptr: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };

    // inputs to test
    let inputs = vec![10, 100, 1000];
    let mut test_cases = Vec::new();

    for &input in &inputs {
        // Run safely in case seed is bad, though unlikely for valid parse
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| func_ptr(input)));

        match result {
            Ok(output) => {
                test_cases.push(TestCase::new(input, output));
                println!("   input={:<5} → expected={:<10} (verified)", input, output);
            }
            Err(_) => {
                println!("❌ Seed code crashed on input {}! Cannot evolve.", input);
                return;
            }
        }
    }
    println!("");

    // Configure evolution
    let config = EvolutionConfig {
        population_size,
        mutation_rate: 0.3,
        crossover_rate: 0.7,
        tournament_size: 5,
        elite_count: 2,
        seed: 42,
    };

    println!("⚙️  Evolution Config:");
    println!("   Population: {}", config.population_size);
    println!("   Generations: {}", generations);
    println!("   Mutation rate: {:.0}%", config.mutation_rate * 100.0);
    println!(
        "   Target speedup: {}",
        target.map_or("None".to_string(), |t| format!("{:.2}x", t))
    );

    // Create evolution engine
    let mut engine = EvolutionEngine::new(seed_function, test_cases, config);

    println!("\n🧬 Starting Evolution...\n");
    println!("┌──────┬────────────────┬────────────────┬────────────────┐");
    println!("│ Gen  │ Best Fitness   │ Valid/Pop      │ Speedup        │");
    println!("├──────┼────────────────┼────────────────┼────────────────┤");

    // Run evolution
    let result = engine.run(generations, target);

    // Display results from history
    for (i, gen_result) in result.history.iter().enumerate() {
        if i < 5 || i % 10 == 0 || i == result.history.len() - 1 {
            let speedup_str = if gen_result.speedup_vs_baseline >= 1.0 {
                format!("✅ {:.2}x", gen_result.speedup_vs_baseline)
            } else {
                format!("   {:.2}x", gen_result.speedup_vs_baseline)
            };

            println!(
                "│ {:4} │ {:>14.0} │ {:>6}/{:<6}  │ {:14} │",
                gen_result.generation,
                gen_result.best_fitness,
                gen_result.valid_count,
                population_size,
                speedup_str
            );
        }
    }
    println!("└──────┴────────────────┴────────────────┴────────────────┘");

    // Final results
    println!("\n{}", "═".repeat(64));
    println!("🏆 EVOLUTION COMPLETE!");
    println!("   Generations run: {}", result.generations_run);
    println!("   Final speedup: {:.2}x", result.final_speedup);
    println!(
        "   Best genome: {} instructions",
        result.best_genome.instructions.len()
    );

    if result.final_speedup > 1.0 {
        println!(
            "\n🎉 Code evolved to be {:.1}% faster than baseline!",
            (result.final_speedup - 1.0) * 100.0
        );
    }

    println!("\n✅ Self-Evolving JIT Complete!\n");
}
