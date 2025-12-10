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
use crate::cpu_features::CpuFeatures;
use crate::parser::Parser;
use crate::variant_generator::VariantGenerator;

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

/// Get NanoForge version
#[pyfunction]
pub fn version() -> &'static str {
    "0.1.0"
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
    Ok(())
}
