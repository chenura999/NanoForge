//! AI-Powered Variant Selection Engine
//!
//! Implements Thompson Sampling and Contextual Bandits for intelligent
//! variant selection based on runtime feedback.

use rand::Rng;
use std::collections::HashMap;

/// Size buckets for contextual decision making
/// The AI learns different policies for different input sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeBucket {
    /// N < 32 - SIMD overhead dominates
    Tiny,
    /// 32 <= N < 256 - Scalar might win
    Small,
    /// 256 <= N < 4096 - AVX starts to win
    Medium,
    /// 4096 <= N < 65536 - AVX clearly better
    Large,
    /// N >= 65536 - Memory bandwidth limited
    Huge,
}

impl SizeBucket {
    /// Classify an input size into a bucket
    pub fn from_size(n: u64) -> Self {
        match n {
            0..=31 => SizeBucket::Tiny,
            32..=255 => SizeBucket::Small,
            256..=4095 => SizeBucket::Medium,
            4096..=65535 => SizeBucket::Large,
            _ => SizeBucket::Huge,
        }
    }

    /// Get all bucket variants for initialization
    pub fn all() -> Vec<SizeBucket> {
        vec![
            SizeBucket::Tiny,
            SizeBucket::Small,
            SizeBucket::Medium,
            SizeBucket::Large,
            SizeBucket::Huge,
        ]
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            SizeBucket::Tiny => "Tiny (<32)",
            SizeBucket::Small => "Small (32-255)",
            SizeBucket::Medium => "Medium (256-4K)",
            SizeBucket::Large => "Large (4K-64K)",
            SizeBucket::Huge => "Huge (>64K)",
        }
    }
}

impl std::fmt::Display for SizeBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Feature vector extracted from runtime context
#[derive(Debug, Clone)]
pub struct OptimizationFeatures {
    /// Input data size (number of elements)
    pub input_size: u64,
    /// Loop trip count (iterations)
    pub loop_trip_count: u64,
    /// Data alignment (0 = unknown, 16, 32, 64)
    pub alignment: u8,
    /// CPU frequency estimate (MHz)
    pub cpu_freq_mhz: u32,
    /// Memory pressure indicator (0.0 - 1.0)
    pub memory_pressure: f32,
}

impl OptimizationFeatures {
    pub fn new(input_size: u64) -> Self {
        Self {
            input_size,
            loop_trip_count: input_size,
            alignment: 0,
            cpu_freq_mhz: 4000, // Assume 4GHz
            memory_pressure: 0.0,
        }
    }

    /// Get the size bucket for this context
    pub fn size_bucket(&self) -> SizeBucket {
        SizeBucket::from_size(self.input_size)
    }

    /// Convert to feature vector for ML
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            (self.input_size as f64).ln(), // Log-scale for size
            (self.loop_trip_count as f64).ln(),
            self.alignment as f64 / 64.0,
            self.cpu_freq_mhz as f64 / 5000.0,
            self.memory_pressure as f64,
        ]
    }
}

impl Default for OptimizationFeatures {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Thompson Sampling Multi-Armed Bandit for variant selection
///
/// Each variant is an "arm" with an unknown success probability.
/// We model each arm with a Beta distribution and sample to select.
#[derive(Debug, Clone)]
pub struct VariantBandit {
    /// Number of variants (arms)
    num_variants: usize,
    /// Success counts (α parameter of Beta distribution)
    successes: Vec<f64>,
    /// Failure counts (β parameter of Beta distribution)
    failures: Vec<f64>,
    /// Variant names for logging
    variant_names: Vec<String>,
    /// Total selections per variant
    selections: Vec<u64>,
}

impl VariantBandit {
    /// Create a new bandit with uniform priors
    pub fn new(variant_names: Vec<String>) -> Self {
        let n = variant_names.len();
        Self {
            num_variants: n,
            successes: vec![1.0; n], // Prior: Beta(1,1) = Uniform
            failures: vec![1.0; n],
            variant_names,
            selections: vec![0; n],
        }
    }

    /// Select a variant using Thompson Sampling
    /// Returns the index of the selected variant
    pub fn select(&mut self) -> usize {
        let mut rng = rand::thread_rng();

        // Sample from each arm's Beta distribution
        let samples: Vec<f64> = self
            .successes
            .iter()
            .zip(&self.failures)
            .map(|(&a, &b)| sample_beta(&mut rng, a, b))
            .collect();

        // Select the arm with highest sample
        let selected = samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.selections[selected] += 1;
        selected
    }

    /// Update beliefs based on benchmark result
    ///
    /// `variant_idx`: The variant that was tested
    /// `was_fastest`: True if this variant was the fastest in the benchmark
    pub fn update(&mut self, variant_idx: usize, was_fastest: bool) {
        if variant_idx >= self.num_variants {
            return;
        }

        if was_fastest {
            self.successes[variant_idx] += 1.0;
        } else {
            self.failures[variant_idx] += 1.0;
        }
    }

    /// Update based on relative performance
    ///
    /// `variant_idx`: The variant that was tested
    /// `cycles`: Cycles per operation achieved
    /// `best_cycles`: Best known cycles per operation
    pub fn update_with_performance(&mut self, variant_idx: usize, cycles: u64, best_cycles: u64) {
        if variant_idx >= self.num_variants {
            return;
        }

        // Calculate relative performance (0.0 = worst, 1.0 = best)
        let performance_ratio = if cycles > 0 {
            best_cycles as f64 / cycles as f64
        } else {
            0.0
        };

        // Update Beta parameters proportionally
        self.successes[variant_idx] += performance_ratio;
        self.failures[variant_idx] += 1.0 - performance_ratio;
    }

    /// Get the current best variant (highest expected value)
    pub fn get_best(&self) -> usize {
        self.successes
            .iter()
            .zip(&self.failures)
            .map(|(a, b)| a / (a + b)) // Expected value of Beta distribution
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get statistics for all variants
    pub fn get_stats(&self) -> Vec<VariantStats> {
        self.variant_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let expected = self.successes[i] / (self.successes[i] + self.failures[i]);
                VariantStats {
                    name: name.clone(),
                    selections: self.selections[i],
                    expected_value: expected,
                    confidence: self.successes[i] + self.failures[i],
                }
            })
            .collect()
    }

    /// Print current state
    pub fn print_status(&self) {
        println!("\n📊 Variant Bandit Status:");
        println!("┌──────────────────────┬───────────┬───────────┬───────────┐");
        println!("│ Variant              │ Selections│ Expected  │ Confidence│");
        println!("├──────────────────────┼───────────┼───────────┼───────────┤");

        for stats in self.get_stats() {
            println!(
                "│ {:20} │ {:9} │ {:9.3} │ {:9.1} │",
                stats.name, stats.selections, stats.expected_value, stats.confidence
            );
        }
        println!("└──────────────────────┴───────────┴───────────┴───────────┘");
    }
}

/// Statistics for a single variant
#[derive(Debug, Clone)]
pub struct VariantStats {
    pub name: String,
    pub selections: u64,
    pub expected_value: f64,
    pub confidence: f64,
}

/// Sample from Beta distribution using rejection sampling
fn sample_beta<R: Rng>(rng: &mut R, alpha: f64, beta: f64) -> f64 {
    // Simple approximation using Gamma distribution
    // Beta(α, β) = Gamma(α, 1) / (Gamma(α, 1) + Gamma(β, 1))
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    x / (x + y)
}

/// Sample from Gamma distribution using Marsaglia and Tsang's method
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        // For shape < 1, use: Gamma(α) = Gamma(α+1) * U^(1/α)
        let u: f64 = rng.gen();
        return sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = sample_normal(rng);
        let v = 1.0 + c * x;

        if v > 0.0 {
            let v = v * v * v;
            let u: f64 = rng.gen();

            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v;
            }

            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

/// Sample from standard normal distribution using Box-Muller
fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================================
// CONTEXTUAL BANDIT - The Key Upgrade for Phase 3
// ============================================================================

/// Contextual Bandit with per-bucket Thompson Sampling
///
/// This is the KEY UPGRADE from the basic bandit:
/// - Maintains a SEPARATE bandit for each SizeBucket
/// - Learns that small inputs → Scalar is better
/// - Learns that large inputs → AVX2 is better
/// - Discovers the decision boundary automatically!
#[derive(Debug)]
pub struct ContextualBandit {
    /// One bandit per size bucket
    bandits: HashMap<SizeBucket, VariantBandit>,
    /// Variant names (shared across all bandits)
    variant_names: Vec<String>,
}

impl ContextualBandit {
    /// Create a new contextual bandit
    pub fn new(variant_names: Vec<String>) -> Self {
        let mut bandits = HashMap::new();

        // Initialize a separate bandit for each size bucket
        for bucket in SizeBucket::all() {
            bandits.insert(bucket, VariantBandit::new(variant_names.clone()));
        }

        Self {
            bandits,
            variant_names,
        }
    }

    /// Select a variant based on context (input size)
    pub fn select(&mut self, context: &OptimizationFeatures) -> usize {
        let bucket = context.size_bucket();
        self.bandits
            .get_mut(&bucket)
            .map(|b| b.select())
            .unwrap_or(0)
    }

    /// Update the bandit for the specific context
    pub fn update(
        &mut self,
        context: &OptimizationFeatures,
        variant_idx: usize,
        was_fastest: bool,
    ) {
        let bucket = context.size_bucket();
        if let Some(bandit) = self.bandits.get_mut(&bucket) {
            bandit.update(variant_idx, was_fastest);
        }
    }

    /// Update with performance ratio for a specific context
    pub fn update_with_performance(
        &mut self,
        context: &OptimizationFeatures,
        variant_idx: usize,
        cycles: u64,
        best_cycles: u64,
    ) {
        let bucket = context.size_bucket();
        if let Some(bandit) = self.bandits.get_mut(&bucket) {
            bandit.update_with_performance(variant_idx, cycles, best_cycles);
        }
    }

    /// Get the best variant for a specific context
    pub fn get_best_for_context(&self, context: &OptimizationFeatures) -> usize {
        let bucket = context.size_bucket();
        self.bandits.get(&bucket).map(|b| b.get_best()).unwrap_or(0)
    }

    /// Get the learned decision boundary as a summary
    pub fn get_decision_boundary(&self) -> Vec<(SizeBucket, String, f64)> {
        let mut decisions = Vec::new();

        for bucket in SizeBucket::all() {
            if let Some(bandit) = self.bandits.get(&bucket) {
                let best_idx = bandit.get_best();
                let stats = bandit.get_stats();
                let best_name = self
                    .variant_names
                    .get(best_idx)
                    .cloned()
                    .unwrap_or_default();
                let expected = stats.get(best_idx).map(|s| s.expected_value).unwrap_or(0.0);
                decisions.push((bucket, best_name, expected));
            }
        }

        decisions
    }

    /// Print the learned decision boundary
    pub fn print_decision_boundary(&self) {
        println!("\n🎯 Learned Decision Boundary:");
        println!("┌──────────────────┬──────────────────┬───────────┐");
        println!("│ Input Size       │ Best Variant     │ Confidence│");
        println!("├──────────────────┼──────────────────┼───────────┤");

        for (bucket, variant, expected) in self.get_decision_boundary() {
            println!(
                "│ {:16} │ {:16} │ {:9.3} │",
                bucket.name(),
                variant,
                expected
            );
        }
        println!("└──────────────────┴──────────────────┴───────────┘");
    }

    /// Print detailed status for all buckets
    pub fn print_full_status(&self) {
        println!("\n📊 Contextual Bandit Full Status:");
        for bucket in SizeBucket::all() {
            if let Some(bandit) = self.bandits.get(&bucket) {
                println!("\n  📦 Bucket: {}", bucket);
                let stats = bandit.get_stats();
                for s in stats {
                    let marker = if s.expected_value > 0.6 { "★" } else { " " };
                    println!(
                        "     {} {:12} exp={:.3} conf={:.1} sel={}",
                        marker, s.name, s.expected_value, s.confidence, s.selections
                    );
                }
            }
        }
    }
}

/// Contextual Bandit with Linear Upper Confidence Bound (LinUCB)
///
/// Uses features to predict which variant will perform best
#[derive(Debug)]
pub struct ContextualSelector {
    /// Number of features
    num_features: usize,
    /// Number of variants
    num_variants: usize,
    /// Weight vectors for each variant
    weights: Vec<Vec<f64>>,
    /// Variant names
    variant_names: Vec<String>,
    /// Exploration parameter
    alpha: f64,
}

impl ContextualSelector {
    pub fn new(variant_names: Vec<String>, num_features: usize) -> Self {
        let n = variant_names.len();
        Self {
            num_features,
            num_variants: n,
            weights: vec![vec![0.0; num_features]; n],
            variant_names,
            alpha: 0.5, // Exploration vs exploitation trade-off
        }
    }

    /// Select variant based on features
    pub fn select(&self, features: &OptimizationFeatures) -> usize {
        let feature_vec = features.to_vector();

        // Compute UCB score for each variant
        let scores: Vec<f64> = self
            .weights
            .iter()
            .map(|w| {
                let expected: f64 = w.iter().zip(&feature_vec).map(|(wi, fi)| wi * fi).sum();

                // Add exploration bonus (simplified)
                expected + self.alpha * (1.0 / (self.num_features as f64).sqrt())
            })
            .collect();

        // Select variant with highest score
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update weights based on observed reward
    pub fn update(&mut self, variant_idx: usize, features: &OptimizationFeatures, reward: f64) {
        if variant_idx >= self.num_variants {
            return;
        }

        let feature_vec = features.to_vector();
        let learning_rate = 0.1;

        // Simple gradient update
        for (i, f) in feature_vec.iter().enumerate() {
            if i < self.num_features {
                self.weights[variant_idx][i] += learning_rate * reward * f;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit_selection() {
        let names = vec![
            "Scalarx1".to_string(),
            "AVX2x2".to_string(),
            "AVX2x4".to_string(),
        ];
        let mut bandit = VariantBandit::new(names);

        // Simulate: AVX2x2 is best
        for _ in 0..100 {
            let selected = bandit.select();
            let was_fastest = selected == 1; // AVX2x2 always wins
            bandit.update(selected, was_fastest);
        }

        // AVX2x2 should have highest expected value
        let best = bandit.get_best();
        println!(
            "Best variant: {} (index {})",
            bandit.variant_names[best], best
        );

        bandit.print_status();

        // Should converge to variant 1 (AVX2x2)
        assert_eq!(best, 1, "Should converge to AVX2x2");
    }

    #[test]
    fn test_contextual_selector() {
        let names = vec!["Scalar".to_string(), "AVX2".to_string()];
        let mut selector = ContextualSelector::new(names, 5);

        let features = OptimizationFeatures::new(10000);
        let selected = selector.select(&features);

        println!("Selected variant: {}", selected);

        // Update with reward
        selector.update(selected, &features, 1.0);
    }
}
