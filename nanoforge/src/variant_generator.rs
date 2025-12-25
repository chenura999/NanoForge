//! Multi-Variant Code Generator
//!
//! Generates multiple optimized variants of the same function using different
//! ISA extensions and optimization strategies. Each variant is benchmarked
//! and the AI optimizer selects the best one for the current workload.

use crate::compiler::Compiler;
use crate::cpu_features::CpuFeatures;
use crate::ir::Program;
use crate::jit_memory::DualMappedMemory;
use crate::optimizer::Optimizer;

/// ISA extension level for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaExtension {
    Scalar,
    Avx2,
    Avx512,
    Amx,
}

impl std::fmt::Display for IsaExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IsaExtension::Scalar => write!(f, "Scalar"),
            IsaExtension::Avx2 => write!(f, "AVX2"),
            IsaExtension::Avx512 => write!(f, "AVX-512"),
            IsaExtension::Amx => write!(f, "AMX"),
        }
    }
}

/// Configuration for a specific variant
#[derive(Debug, Clone)]
pub struct VariantConfig {
    pub isa: IsaExtension,
    pub unroll_factor: u8,
    pub optimization_level: u8,
    pub name: String,
}

impl VariantConfig {
    pub fn new(isa: IsaExtension, unroll_factor: u8, opt_level: u8) -> Self {
        let name = format!("{}x{}", isa, unroll_factor);
        Self {
            isa,
            unroll_factor,
            optimization_level: opt_level,
            name,
        }
    }
}

/// A compiled variant ready for execution and benchmarking
#[derive(Debug)]
pub struct CompiledVariant {
    pub config: VariantConfig,
    pub memory: DualMappedMemory,
    pub code_size: usize,
    pub entry_offset: usize,
    pub func_ptr: extern "C" fn(u64) -> u64,
}

impl CompiledVariant {
    /// Execute this variant with the given input
    pub fn execute(&self, input: u64) -> u64 {
        (self.func_ptr)(input)
    }
}

/// Generates multiple code variants for a function
pub struct VariantGenerator {
    cpu_features: CpuFeatures,
}

impl VariantGenerator {
    pub fn new() -> Self {
        Self {
            cpu_features: CpuFeatures::detect(),
        }
    }

    pub fn with_features(features: CpuFeatures) -> Self {
        Self {
            cpu_features: features,
        }
    }

    /// Generate all viable variant configurations for the current CPU
    pub fn get_variant_configs(&self) -> Vec<VariantConfig> {
        let mut configs = vec![];

        // Always include scalar baseline
        configs.push(VariantConfig::new(IsaExtension::Scalar, 1, 1));
        configs.push(VariantConfig::new(IsaExtension::Scalar, 2, 2));
        configs.push(VariantConfig::new(IsaExtension::Scalar, 4, 2));
        // High Register Pressure Stress Test
        configs.push(VariantConfig::new(IsaExtension::Scalar, 8, 2));
        configs.push(VariantConfig::new(IsaExtension::Scalar, 16, 2));

        // AVX2 variants (if supported)
        if self.cpu_features.has_avx2() {
            configs.push(VariantConfig::new(IsaExtension::Avx2, 2, 3));
            configs.push(VariantConfig::new(IsaExtension::Avx2, 4, 3));
            configs.push(VariantConfig::new(IsaExtension::Avx2, 8, 3));
        }

        // AVX-512 variants (if supported)
        if self.cpu_features.has_avx512() {
            configs.push(VariantConfig::new(IsaExtension::Avx512, 4, 3));
            configs.push(VariantConfig::new(IsaExtension::Avx512, 8, 3));
            configs.push(VariantConfig::new(IsaExtension::Avx512, 16, 3));
        }

        // AMX variants (if supported)
        if self.cpu_features.has_amx() {
            configs.push(VariantConfig::new(IsaExtension::Amx, 1, 3));
        }

        configs
    }

    /// Generate all viable variants for a program
    pub fn generate_variants(&self, program: &Program) -> Result<Vec<CompiledVariant>, String> {
        let configs = self.get_variant_configs();
        let mut variants = Vec::with_capacity(configs.len());

        for config in configs {
            match self.compile_variant(program, &config) {
                Ok(variant) => variants.push(variant),
                Err(e) => {
                    // Log but continue - some variants may fail
                    tracing::warn!("Failed to compile variant {}: {}", config.name, e);
                }
            }
        }

        if variants.is_empty() {
            return Err("Failed to compile any variants".to_string());
        }

        Ok(variants)
    }

    /// Compile a specific variant
    fn compile_variant(
        &self,
        program: &Program,
        config: &VariantConfig,
    ) -> Result<CompiledVariant, String> {
        // Clone the program for optimization
        let mut prog = program.clone();

        // Apply optimization based on config
        let opt_level = match config.isa {
            IsaExtension::Scalar => config.optimization_level.min(2),
            IsaExtension::Avx2 => 3, // Force vectorization
            IsaExtension::Avx512 => 3,
            IsaExtension::Amx => 3,
        };

        Optimizer::optimize_program(&mut prog, opt_level);

        // Compile to machine code
        let (code, entry_offset) = Compiler::compile_program(&prog, opt_level)?;
        let code_size = code.len();

        // Allocate executable memory
        let memory = DualMappedMemory::new(code_size.max(4096))?;

        // Copy code to memory
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), memory.rw_ptr, code_size);
        }
        memory.flush_icache();

        // Create function pointer
        let func_ptr: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(memory.rx_ptr.add(entry_offset)) };

        Ok(CompiledVariant {
            config: config.clone(),
            memory,
            code_size,
            entry_offset,
            func_ptr,
        })
    }

    /// Get detected CPU features
    pub fn cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }
}

impl Default for VariantGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    #[test]
    fn test_variant_generation() {
        let source = r#"
            fn main() {
                x = 42
                y = x + 10
                return y
            }
        "#;

        let mut parser = Parser::new();
        let program = parser.parse(source).expect("Parse failed");

        let generator = VariantGenerator::new();
        let configs = generator.get_variant_configs();

        println!("CPU: {}", generator.cpu_features().summary());
        println!("Generated {} variant configs:", configs.len());
        for config in &configs {
            println!("  - {}", config.name);
        }

        assert!(!configs.is_empty());
    }
}
