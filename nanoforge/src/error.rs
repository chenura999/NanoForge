//! Error Types for NanoForge
//!
//! Provides a unified error type for all NanoForge operations,
//! replacing panics with proper error handling.

use std::fmt;

/// Unified error type for NanoForge operations
#[derive(Debug, Clone)]
pub enum NanoForgeError {
    /// Failed to parse the NanoForge script
    ParseError(String),
    /// Failed to compile the program
    CompileError(String),
    /// Memory allocation failed
    MemoryError(String),
    /// JIT execution failed
    ExecutionError(String),
    /// AI optimizer error
    OptimizerError(String),
    /// I/O operation failed
    IoError(String),
    /// Security violation
    SecurityError(String),
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
    /// Invalid configuration
    ConfigError(String),
}

impl fmt::Display for NanoForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NanoForgeError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            NanoForgeError::CompileError(msg) => write!(f, "Compile error: {}", msg),
            NanoForgeError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            NanoForgeError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            NanoForgeError::OptimizerError(msg) => write!(f, "Optimizer error: {}", msg),
            NanoForgeError::IoError(msg) => write!(f, "I/O error: {}", msg),
            NanoForgeError::SecurityError(msg) => write!(f, "Security error: {}", msg),
            NanoForgeError::ResourceLimitExceeded(msg) => {
                write!(f, "Resource limit exceeded: {}", msg)
            }
            NanoForgeError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for NanoForgeError {}

/// Result type alias for NanoForge operations
pub type Result<T> = std::result::Result<T, NanoForgeError>;

// Conversion from std::io::Error
impl From<std::io::Error> for NanoForgeError {
    fn from(err: std::io::Error) -> Self {
        NanoForgeError::IoError(err.to_string())
    }
}

// Conversion from serde_json::Error
impl From<serde_json::Error> for NanoForgeError {
    fn from(err: serde_json::Error) -> Self {
        NanoForgeError::IoError(format!("JSON error: {}", err))
    }
}

/// Security limits for script execution
#[derive(Debug, Clone)]
pub struct SecurityLimits {
    /// Maximum script source size in bytes
    pub max_script_size: usize,
    /// Maximum generated code size in bytes
    pub max_code_size: usize,
    /// Maximum memory allocation in bytes
    pub max_memory: usize,
    /// Maximum number of instructions
    pub max_instructions: usize,
    /// Maximum execution time in milliseconds
    pub max_execution_ms: u64,
    /// Maximum loop iterations
    pub max_loop_iterations: u64,
}

impl Default for SecurityLimits {
    fn default() -> Self {
        Self {
            max_script_size: 1024 * 1024,   // 1 MB
            max_code_size: 1024 * 1024,     // 1 MB
            max_memory: 256 * 1024 * 1024,  // 256 MB
            max_instructions: 10_000,       // 10K instructions
            max_execution_ms: 5000,         // 5 seconds
            max_loop_iterations: 1_000_000, // 1M iterations
        }
    }
}

impl SecurityLimits {
    /// Create strict limits for untrusted code
    pub fn strict() -> Self {
        Self {
            max_script_size: 64 * 1024,   // 64 KB
            max_code_size: 256 * 1024,    // 256 KB
            max_memory: 16 * 1024 * 1024, // 16 MB
            max_instructions: 1000,       // 1K instructions
            max_execution_ms: 1000,       // 1 second
            max_loop_iterations: 100_000, // 100K iterations
        }
    }

    /// Create relaxed limits for trusted code
    pub fn trusted() -> Self {
        Self {
            max_script_size: 10 * 1024 * 1024,  // 10 MB
            max_code_size: 10 * 1024 * 1024,    // 10 MB
            max_memory: 1024 * 1024 * 1024,     // 1 GB
            max_instructions: 1_000_000,        // 1M instructions
            max_execution_ms: 60_000,           // 60 seconds
            max_loop_iterations: 1_000_000_000, // 1B iterations
        }
    }

    /// Check if script size is within limits
    pub fn check_script_size(&self, size: usize) -> Result<()> {
        if size > self.max_script_size {
            return Err(NanoForgeError::ResourceLimitExceeded(format!(
                "Script size {} bytes exceeds limit {} bytes",
                size, self.max_script_size
            )));
        }
        Ok(())
    }

    /// Check if code size is within limits
    pub fn check_code_size(&self, size: usize) -> Result<()> {
        if size > self.max_code_size {
            return Err(NanoForgeError::ResourceLimitExceeded(format!(
                "Code size {} bytes exceeds limit {} bytes",
                size, self.max_code_size
            )));
        }
        Ok(())
    }

    /// Check if instruction count is within limits
    pub fn check_instruction_count(&self, count: usize) -> Result<()> {
        if count > self.max_instructions {
            return Err(NanoForgeError::ResourceLimitExceeded(format!(
                "Instruction count {} exceeds limit {}",
                count, self.max_instructions
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NanoForgeError::ParseError("unexpected token".to_string());
        assert!(err.to_string().contains("Parse error"));
    }

    #[test]
    fn test_security_limits_default() {
        let limits = SecurityLimits::default();
        assert!(limits.check_script_size(1000).is_ok());
        assert!(limits.check_script_size(10 * 1024 * 1024).is_err());
    }

    #[test]
    fn test_security_limits_strict() {
        let limits = SecurityLimits::strict();
        assert!(limits.check_script_size(1000).is_ok());
        assert!(limits.check_script_size(100 * 1024).is_err());
    }
}
