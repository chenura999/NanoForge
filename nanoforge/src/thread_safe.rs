//! Thread-Safe Wrapper for NanoForge AI Optimizer
//!
//! Provides a thread-safe wrapper around the ContextualBandit
//! using Mutex for safe concurrent access.

use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use crate::ai_optimizer::{ContextualBandit, OptimizationFeatures, SizeBucket};
use crate::error::{NanoForgeError, Result};

/// Thread-safe AI optimizer wrapper
///
/// Uses RwLock for read-heavy workloads (selecting variants)
/// and Mutex for writes (updating with feedback).
#[derive(Clone)]
pub struct ThreadSafeOptimizer {
    inner: Arc<RwLock<ContextualBandit>>,
    variant_names: Arc<Vec<String>>,
}

impl ThreadSafeOptimizer {
    /// Create a new thread-safe optimizer
    pub fn new(variant_names: Vec<String>) -> Self {
        let bandit = ContextualBandit::new(variant_names.clone());
        Self {
            inner: Arc::new(RwLock::new(bandit)),
            variant_names: Arc::new(variant_names),
        }
    }

    /// Load from file or create new (thread-safe)
    pub fn load_or_new(path: &Path, variant_names: Vec<String>) -> Self {
        let bandit = ContextualBandit::load_or_new(path, variant_names.clone());
        Self {
            inner: Arc::new(RwLock::new(bandit)),
            variant_names: Arc::new(variant_names),
        }
    }

    /// Select variant (read lock, allows concurrent reads)
    pub fn select(&self, input_size: u64) -> Result<usize> {
        let features = OptimizationFeatures::new(input_size);
        let mut guard = self
            .inner
            .write()
            .map_err(|e| NanoForgeError::OptimizerError(format!("Lock poisoned: {}", e)))?;
        Ok(guard.select(&features))
    }

    /// Update with performance (write lock, exclusive access)
    pub fn update(
        &self,
        input_size: u64,
        variant_idx: usize,
        cycles: u64,
        best_cycles: u64,
    ) -> Result<()> {
        let features = OptimizationFeatures::new(input_size);
        let mut guard = self
            .inner
            .write()
            .map_err(|e| NanoForgeError::OptimizerError(format!("Lock poisoned: {}", e)))?;
        guard.update_with_performance(&features, variant_idx, cycles, best_cycles);
        Ok(())
    }

    /// Save to file (read lock)
    pub fn save(&self, path: &Path) -> Result<()> {
        let guard = self
            .inner
            .read()
            .map_err(|e| NanoForgeError::OptimizerError(format!("Lock poisoned: {}", e)))?;
        guard
            .save_to_file(path)
            .map_err(|e| NanoForgeError::IoError(e))
    }

    /// Get decision boundary (read lock)
    pub fn get_decision_boundary(&self) -> Result<Vec<(SizeBucket, String, f64)>> {
        let guard = self
            .inner
            .read()
            .map_err(|e| NanoForgeError::OptimizerError(format!("Lock poisoned: {}", e)))?;
        Ok(guard.get_decision_boundary())
    }

    /// Get variant names
    pub fn variant_names(&self) -> &[String] {
        &self.variant_names
    }

    /// Get current best for a context (read lock)
    pub fn get_best_for_size(&self, input_size: u64) -> Result<usize> {
        let features = OptimizationFeatures::new(input_size);
        let guard = self
            .inner
            .read()
            .map_err(|e| NanoForgeError::OptimizerError(format!("Lock poisoned: {}", e)))?;
        Ok(guard.get_best_for_context(&features))
    }
}

// Implement Send + Sync for ThreadSafeOptimizer
unsafe impl Send for ThreadSafeOptimizer {}
unsafe impl Sync for ThreadSafeOptimizer {}

/// Atomic optimizer guard for scoped operations
pub struct OptimizerGuard<'a> {
    inner: std::sync::RwLockWriteGuard<'a, ContextualBandit>,
}

impl<'a> OptimizerGuard<'a> {
    /// Select a variant
    pub fn select(&mut self, input_size: u64) -> usize {
        let features = OptimizationFeatures::new(input_size);
        self.inner.select(&features)
    }

    /// Update with feedback
    pub fn update(&mut self, input_size: u64, variant_idx: usize, cycles: u64, best_cycles: u64) {
        let features = OptimizationFeatures::new(input_size);
        self.inner
            .update_with_performance(&features, variant_idx, cycles, best_cycles);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_thread_safe_optimizer() {
        let variants = vec!["Scalar".to_string(), "AVX2".to_string()];
        let opt = ThreadSafeOptimizer::new(variants);

        // Test concurrent reads
        let opt1 = opt.clone();
        let opt2 = opt.clone();

        let h1 = thread::spawn(move || {
            for _ in 0..100 {
                let _ = opt1.select(100);
            }
        });

        let h2 = thread::spawn(move || {
            for _ in 0..100 {
                let _ = opt2.select(10000);
            }
        });

        h1.join().unwrap();
        h2.join().unwrap();
    }

    #[test]
    fn test_concurrent_updates() {
        let variants = vec!["A".to_string(), "B".to_string()];
        let opt = ThreadSafeOptimizer::new(variants);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let opt = opt.clone();
                thread::spawn(move || {
                    for j in 0..25 {
                        let _ = opt.update(100 * (i as u64 + 1), j % 2, 100, 80);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should complete without panicking
        assert!(opt.get_decision_boundary().is_ok());
    }
}
