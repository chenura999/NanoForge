//! Validator for Evolved Code
//!
//! Ensures that mutated/evolved code produces correct results
//! and doesn't crash or hang.

use crate::compiler::Compiler;
use crate::ir::Program;
use crate::jit_memory::DualMappedMemory;
use crate::mutator::Genome;
use std::time::{Duration, Instant};

/// Result of validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Code is valid and produces correct output
    Valid { output: i64, execution_time_ns: u64 },
    /// Code produces wrong output
    WrongOutput { expected: i64, actual: i64 },
    /// Code took too long (timeout)
    Timeout,
    /// Code failed to compile
    CompileError(String),
    /// Code crashed during execution
    Crashed,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid { .. })
    }
}

/// Test case for validation
#[derive(Debug, Clone)]
pub struct TestCase {
    pub input: i64,
    pub expected_output: i64,
}

impl TestCase {
    pub fn new(input: i64, expected_output: i64) -> Self {
        Self {
            input,
            expected_output,
        }
    }
}

/// Validator configuration
pub struct ValidatorConfig {
    /// Maximum execution time per test case
    pub timeout: Duration,
    /// Number of warmup runs before timing
    pub warmup_runs: u32,
    /// Number of timing runs for averaging
    pub timing_runs: u32,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(100),
            warmup_runs: 2,
            timing_runs: 5,
        }
    }
}

/// Validator for evolved genomes
pub struct Validator {
    config: ValidatorConfig,
}

impl Validator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Validate a genome against test cases
    pub fn validate(&self, genome: &Genome, test_cases: &[TestCase]) -> ValidationResult {
        // Convert genome to function
        let func = genome.to_function();

        // Create program with single function
        let mut program = Program::new();
        program.add_function(func);

        // Compile to machine code - wrapped in catch_unwind because
        // mutated genomes might cause panics in the assembler (e.g., missing labels)
        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Compiler::compile_program(&program, 0)
        }));

        let (code, _) = match compile_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return ValidationResult::CompileError(e),
            Err(_) => {
                return ValidationResult::CompileError(
                    "Compilation panicked (invalid genome)".to_string(),
                )
            }
        };

        // Allocate executable memory
        let memory = match DualMappedMemory::new(code.len().max(4096)) {
            Ok(m) => m,
            Err(e) => {
                return ValidationResult::CompileError(format!("Memory allocation failed: {}", e))
            }
        };

        // Copy code to memory
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), memory.rw_ptr, code.len());
        }
        memory.flush_icache();

        // Create function pointer
        let func_ptr: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };

        // Run test cases
        let mut total_time_ns: u64 = 0;
        let mut test_count = 0;

        for test_case in test_cases {
            // Execute with timeout protection
            match self.execute_with_timeout(func_ptr, test_case.input) {
                ExecutionResult::Success(output, time_ns) => {
                    if output != test_case.expected_output {
                        return ValidationResult::WrongOutput {
                            expected: test_case.expected_output,
                            actual: output,
                        };
                    }
                    total_time_ns += time_ns;
                    test_count += 1;
                }
                ExecutionResult::Timeout => return ValidationResult::Timeout,
                ExecutionResult::Crashed => return ValidationResult::Crashed,
            }
        }

        let avg_time_ns = if test_count > 0 {
            total_time_ns / test_count as u64
        } else {
            0
        };

        ValidationResult::Valid {
            output: test_cases.last().map(|tc| tc.expected_output).unwrap_or(0),
            execution_time_ns: avg_time_ns,
        }
    }

    /// Execute function with timeout protection
    fn execute_with_timeout(&self, func: extern "C" fn(i64) -> i64, input: i64) -> ExecutionResult {
        // Warmup runs (no timing)
        for _ in 0..self.config.warmup_runs {
            // TODO: Add actual timeout using signals/threads for production
            // For now, just execute directly (assumes code won't infinite loop)
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| func(input)));
        }

        // Timed runs
        let mut total_ns: u64 = 0;
        let mut last_output: i64 = 0;

        for _ in 0..self.config.timing_runs {
            let start = Instant::now();

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| func(input)));

            let elapsed = start.elapsed();

            match result {
                Ok(output) => {
                    last_output = output;
                    total_ns += elapsed.as_nanos() as u64;
                }
                Err(_) => {
                    return ExecutionResult::Crashed;
                }
            }

            // Check timeout
            if elapsed > self.config.timeout {
                return ExecutionResult::Timeout;
            }
        }

        let avg_ns = total_ns / self.config.timing_runs as u64;
        ExecutionResult::Success(last_output, avg_ns)
    }

    /// Validate and return fitness score (lower is better)
    pub fn fitness(&self, genome: &Genome, test_cases: &[TestCase]) -> Option<f64> {
        match self.validate(genome, test_cases) {
            ValidationResult::Valid {
                execution_time_ns, ..
            } => Some(execution_time_ns as f64),
            _ => None, // Invalid genomes have no fitness
        }
    }
}

/// Result of a single execution attempt
enum ExecutionResult {
    Success(i64, u64), // (output, time_ns)
    Timeout,
    Crashed,
}

impl Default for Validator {
    fn default() -> Self {
        Self::new(ValidatorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Instruction, Opcode, Operand};

    fn create_simple_genome() -> Genome {
        // Simple function: return input + 1
        Genome {
            instructions: vec![
                Instruction {
                    op: Opcode::LoadArg(0),
                    dest: Some(Operand::Reg(0)),
                    src1: None,
                    src2: None,
                },
                Instruction {
                    op: Opcode::Add,
                    dest: Some(Operand::Reg(0)),
                    src1: Some(Operand::Imm(1)),
                    src2: None,
                },
                Instruction {
                    op: Opcode::Ret,
                    dest: Some(Operand::Reg(0)),
                    src1: None,
                    src2: None,
                },
            ],
            name: "add_one".to_string(),
            args: vec!["x".to_string()],
            fitness: None,
            generation: 0,
        }
    }

    #[test]
    fn test_validation_result() {
        let valid = ValidationResult::Valid {
            output: 42,
            execution_time_ns: 1000,
        };
        assert!(valid.is_valid());

        let wrong = ValidationResult::WrongOutput {
            expected: 42,
            actual: 41,
        };
        assert!(!wrong.is_valid());
    }

    #[test]
    fn test_test_case() {
        let tc = TestCase::new(10, 11);
        assert_eq!(tc.input, 10);
        assert_eq!(tc.expected_output, 11);
    }
}
