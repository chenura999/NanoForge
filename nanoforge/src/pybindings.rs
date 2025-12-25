//! Python Bindings for NanoForge
//!
//! Exposes NanoForge to Python using PyO3.
//! Build with: `maturin develop --features python`
//!
//! Example usage in Python:
//! ```python
//! import nanoforge
//!
//! # Check CPU features
//! print(nanoforge.cpu_features())
//!
//! # Create AI optimizer
//! opt = nanoforge.Optimizer()
//! variant = opt.select(input_size=10000)
//! opt.update(input_size=10000, variant_idx=variant, cycles=1000, best_cycles=800)
//! opt.save("brain.json")
//! ```

#![cfg(feature = "python")]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;

use crate::ai_optimizer::{ContextualBandit, OptimizationFeatures, SizeBucket};
use crate::array_ops;
use crate::cpu_features::CpuFeatures;
use crate::parser::Parser;
use crate::variant_generator::VariantGenerator;

use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use std::time::Instant;

/// Python-exposed AI Optimizer using Contextual Bandit
#[pyclass]
pub struct Optimizer {
    bandit: ContextualBandit,
    variant_names: Vec<String>,
}

#[pymethods]
impl Optimizer {
    /// Create a new optimizer
    #[new]
    pub fn new() -> Self {
        let variant_names = vec![
            "Scalarx1".to_string(),
            "Scalarx2".to_string(),
            "Scalarx4".to_string(),
            "AVX2x2".to_string(),
            "AVX2x4".to_string(),
            "AVX2x8".to_string(),
        ];
        let bandit = ContextualBandit::new(variant_names.clone());
        Self {
            bandit,
            variant_names,
        }
    }

    /// Load optimizer from file (or create new if not exists)
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let variant_names = vec![
            "Scalarx1".to_string(),
            "Scalarx2".to_string(),
            "Scalarx4".to_string(),
            "AVX2x2".to_string(),
            "AVX2x4".to_string(),
            "AVX2x8".to_string(),
        ];
        let bandit = ContextualBandit::load_or_new(Path::new(path), variant_names.clone());
        Ok(Self {
            bandit,
            variant_names,
        })
    }

    /// Select the best variant for the given input size
    pub fn select(&mut self, input_size: u64) -> usize {
        let features = OptimizationFeatures::new(input_size);
        self.bandit.select(&features)
    }

    /// Update optimizer with performance feedback
    pub fn update(&mut self, input_size: u64, variant_idx: usize, cycles: u64, best_cycles: u64) {
        let features = OptimizationFeatures::new(input_size);
        self.bandit
            .update_with_performance(&features, variant_idx, cycles, best_cycles);
    }

    /// Save optimizer state to file
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.bandit
            .save_to_file(Path::new(path))
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Get the current best variant for each size bucket
    pub fn get_decision_boundary(&self) -> Vec<(String, String, f64)> {
        self.bandit
            .get_decision_boundary()
            .into_iter()
            .map(|(bucket, variant, confidence)| (bucket.name().to_string(), variant, confidence))
            .collect()
    }

    /// Get variant names
    pub fn variant_names(&self) -> Vec<String> {
        self.variant_names.clone()
    }

    /// Get the size bucket for an input size
    pub fn get_bucket(&self, input_size: u64) -> String {
        SizeBucket::from_size(input_size).name().to_string()
    }
}

/// Python-exposed compiled function
/// Stores the full CompiledVariant to keep the JIT memory alive
#[pyclass]
pub struct CompiledFunction {
    // Keep the variant alive to prevent the JIT memory from being freed
    #[allow(dead_code)]
    variant: crate::variant_generator::CompiledVariant,
}

#[pymethods]
impl CompiledFunction {
    /// Execute the function with the given input
    pub fn execute(&self, input: u64) -> u64 {
        self.variant.execute(input)
    }

    /// Call the function (alias for execute)
    pub fn __call__(&self, input: u64) -> u64 {
        self.execute(input)
    }

    /// Get the variant name
    pub fn name(&self) -> String {
        self.variant.config.name.clone()
    }
}

/// Get CPU features as a string
#[pyfunction]
pub fn cpu_features() -> String {
    CpuFeatures::detect().summary()
}

/// Get detailed CPU feature detection
#[pyfunction]
pub fn cpu_info() -> std::collections::HashMap<String, bool> {
    let features = CpuFeatures::detect();
    let mut map = std::collections::HashMap::new();
    map.insert("sse2".to_string(), features.has_sse2);
    map.insert("sse4_1".to_string(), features.has_sse4_1);
    map.insert("sse4_2".to_string(), features.has_sse4_2);
    map.insert("avx".to_string(), features.has_avx);
    map.insert("avx2".to_string(), features.has_avx2);
    map.insert("avx512f".to_string(), features.has_avx512f);
    map.insert("amx_tile".to_string(), features.has_amx_tile);
    map
}

/// Compile a NanoForge script
#[pyfunction]
pub fn compile(source: &str) -> PyResult<CompiledFunction> {
    let mut parser = Parser::new();
    let program = parser
        .parse(source)
        .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

    let generator = VariantGenerator::new();
    let mut variants = generator
        .generate_variants(&program)
        .map_err(|e| PyValueError::new_err(format!("Compile error: {}", e)))?;

    if variants.is_empty() {
        return Err(PyValueError::new_err("No variants generated"));
    }

    // Take ownership of the first variant
    let variant = variants.remove(0);

    Ok(CompiledFunction { variant })
}

// ============================================================================
// NumPy Array Operations (Zero-Copy, AVX2 Accelerated)
// ============================================================================

/// Add two arrays: C = A + B (AVX2 accelerated)
///
/// Example:
/// ```python
/// import numpy as np
/// import nanoforge
/// a = np.array([1, 2, 3, 4], dtype=np.int64)
/// b = np.array([10, 20, 30, 40], dtype=np.int64)
/// c = np.zeros(4, dtype=np.int64)
/// nanoforge.vec_add(a, b, c)
/// print(c)  # [11, 22, 33, 44]
/// ```
#[pyfunction]
pub fn vec_add<'py>(
    a: PyReadonlyArray1<'py, i64>,
    b: PyReadonlyArray1<'py, i64>,
    c: &PyArray1<i64>,
) -> PyResult<()> {
    let a_slice = a
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array a not contiguous: {}", e)))?;
    let b_slice = b
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array b not contiguous: {}", e)))?;

    // Get mutable slice from c
    let c_slice = unsafe { c.as_slice_mut() }
        .map_err(|e| PyValueError::new_err(format!("Array c not contiguous: {}", e)))?;

    if a_slice.len() != b_slice.len() || a_slice.len() != c_slice.len() {
        return Err(PyValueError::new_err(format!(
            "Array size mismatch: a={}, b={}, c={}",
            a_slice.len(),
            b_slice.len(),
            c_slice.len()
        )));
    }

    array_ops::vec_add_i64(a_slice, b_slice, c_slice);
    Ok(())
}

/// Sum all elements of an array (AVX2 accelerated)
///
/// Example:
/// ```python
/// import numpy as np
/// import nanoforge
/// arr = np.arange(1000000, dtype=np.int64)
/// total = nanoforge.vec_sum(arr)
/// ```
#[pyfunction]
pub fn vec_sum(arr: PyReadonlyArray1<i64>) -> PyResult<i64> {
    let slice = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {}", e)))?;
    Ok(array_ops::vec_sum_i64(slice))
}

/// Scale array in-place: arr *= scalar
#[pyfunction]
pub fn vec_scale(mut arr: PyReadwriteArray1<i64>, scalar: i64) -> PyResult<()> {
    let slice = arr
        .as_slice_mut()
        .map_err(|e| PyValueError::new_err(format!("Array not contiguous: {}", e)))?;
    array_ops::vec_scale_i64(slice, scalar);
    Ok(())
}

/// Benchmark vec_add: returns (nanoforge_ns, numpy_estimated_ns)
/// This runs NanoForge vec_add and estimates NumPy time based on memory bandwidth
#[pyfunction]
pub fn benchmark_vec_add(py: Python<'_>, size: usize) -> PyResult<(u64, u64)> {
    // Create test arrays
    let a: Vec<i64> = (0..size as i64).collect();
    let b: Vec<i64> = (0..size as i64).map(|x| x * 2).collect();
    let mut c = vec![0i64; size];

    // Warmup
    array_ops::vec_add_i64(&a, &b, &mut c);

    // Benchmark NanoForge
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        array_ops::vec_add_i64(&a, &b, &mut c);
    }
    let nanoforge_ns = start.elapsed().as_nanos() as u64 / iterations;

    // Estimate NumPy time (run actual NumPy via Python)
    let numpy_ns = py
        .eval(
            &format!(
                r#"
import numpy as np
import time
a = np.arange({}, dtype=np.int64)
b = np.arange({}, dtype=np.int64) * 2
c = np.zeros({}, dtype=np.int64)
start = time.perf_counter_ns()
for _ in range(100):
    np.add(a, b, out=c)
int((time.perf_counter_ns() - start) / 100)
"#,
                size, size, size
            ),
            None,
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("NumPy benchmark failed: {}", e)))?
        .extract::<u64>()
        .unwrap_or(0);

    Ok((nanoforge_ns, numpy_ns))
}

/// Get NanoForge version
#[pyfunction]
pub fn version() -> &'static str {
    "0.1.0"
}

#[pyfunction]
pub fn evolve(script: String, generations: u32, population: usize) -> PyResult<(String, f64)> {
    use crate::assembler::CodeGenerator;
    use crate::compiler::Compiler;
    use crate::evolution::{EvolutionConfig, EvolutionEngine};
    use crate::jit_memory::DualMappedMemory;
    use crate::validator::TestCase;

    let mut parser = Parser::new();
    let program = parser
        .parse(&script)
        .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

    if program.functions.is_empty() {
        return Err(PyValueError::new_err("No functions found"));
    }

    let seed_function = &program.functions[0];
    println!("🌱 Seed function: {}", seed_function.name);

    // --- Generate Ground Truth ---
    println!("🧪 Generating Ground Truth from Seed Code...");

    // Compile seed to run it
    let (code, main_offset) = Compiler::compile_program(&program, 0)
        .map_err(|e| PyValueError::new_err(format!("Compile error: {}", e)))?;

    let memory = DualMappedMemory::new(code.len() + 4096)
        .map_err(|_| PyValueError::new_err("Memory alloc failed"))?;
    CodeGenerator::emit_to_memory(&memory, &code, 0);

    // Cast to function pointer
    let func_ptr: extern "C" fn(i64) -> i64 =
        unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };

    // inputs to test
    let inputs = vec![10, 100, 1000];
    let mut test_cases = Vec::new();

    for &input in &inputs {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| func_ptr(input)));

        match result {
            Ok(output) => {
                test_cases.push(TestCase::new(input, output));
                println!("   input={:<5} → expected={:<10} (verified)", input, output);
            }
            Err(_) => {
                return Err(PyValueError::new_err(format!(
                    "Seed code crashed on input {}",
                    input
                )));
            }
        }
    }

    // Configure evolution
    let config = EvolutionConfig {
        population_size: population,
        mutation_rate: 0.3,
        crossover_rate: 0.7,
        tournament_size: 5,
        elite_count: 2,
        seed: 42,
    };

    // Create evolution engine
    let mut engine = EvolutionEngine::new(seed_function, test_cases, config);

    println!("\n🧬 Starting Evolution...\n");
    let result = engine.run(generations, None);

    // TODO: Convert best genome to string representation
    let best_code = format!(
        "// Best genome: {} instructions\n// Speedup: {:.2}x\n",
        result.best_genome.instructions.len(),
        result.final_speedup
    );

    Ok((best_code, result.final_speedup))
}

/// Python module definition
#[pymodule]
fn nanoforge(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Optimizer>()?;
    m.add_class::<CompiledFunction>()?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    // NumPy array operations
    m.add_function(wrap_pyfunction!(vec_add, m)?)?;
    m.add_function(wrap_pyfunction!(vec_sum, m)?)?;
    m.add_function(wrap_pyfunction!(vec_scale, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_vec_add, m)?)?;
    // Evolution
    m.add_function(wrap_pyfunction!(evolve, m)?)?;
    Ok(())
}
