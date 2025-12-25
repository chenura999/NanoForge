//! Genetic Algorithm Evolution Engine
//!
//! Core engine that evolves code populations through selection,
//! crossover, and mutation to discover optimal implementations.

use crate::ir::Function;
use crate::mutator::{Genome, Mutator};
use crate::validator::{TestCase, Validator, ValidatorConfig};
use rand::prelude::*;

/// Configuration for the evolution process
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Number of genomes in the population
    pub population_size: usize,
    /// Probability of mutation (0.0 - 1.0)
    pub mutation_rate: f64,
    /// Probability of crossover (0.0 - 1.0)
    pub crossover_rate: f64,
    /// Number of genomes in tournament selection
    pub tournament_size: usize,
    /// Elite count (best genomes preserved unchanged)
    pub elite_count: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            mutation_rate: 0.3,
            crossover_rate: 0.7,
            tournament_size: 5,
            elite_count: 2,
            seed: 42,
        }
    }
}

/// Result of a single generation's evolution
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub generation: u32,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub valid_count: usize,
    pub speedup_vs_baseline: f64,
}

/// Result of the evolution process
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub best_genome: Genome,
    pub generations_run: u32,
    pub final_speedup: f64,
    pub history: Vec<GenerationResult>,
}

/// The main evolution engine
pub struct EvolutionEngine {
    /// Current population of genomes
    population: Vec<Genome>,
    /// The best genome ever seen
    best_ever: Option<Genome>,
    /// Baseline fitness (original code's performance)
    baseline_fitness: f64,
    /// Current generation number
    generation: u32,
    /// Configuration
    config: EvolutionConfig,
    /// Mutator for genetic operations
    mutator: Mutator,
    /// Validator for fitness evaluation
    validator: Validator,
    /// Test cases for validation
    test_cases: Vec<TestCase>,
    /// RNG for selection
    rng: StdRng,
    /// History of generation results
    history: Vec<GenerationResult>,
}

impl EvolutionEngine {
    /// Create a new evolution engine with a seed genome
    pub fn new(
        seed_function: &Function,
        test_cases: Vec<TestCase>,
        config: EvolutionConfig,
    ) -> Self {
        let seed_genome = Genome::from_function(seed_function);
        let mutator = Mutator::new(config.mutation_rate, config.seed);
        let validator = Validator::new(ValidatorConfig::default());
        let rng = StdRng::seed_from_u64(config.seed);

        // Initialize population with copies of seed (will be mutated)
        let population: Vec<Genome> = (0..config.population_size)
            .map(|_| seed_genome.clone())
            .collect();

        Self {
            population,
            best_ever: None,
            baseline_fitness: f64::MAX,
            generation: 0,
            config,
            mutator,
            validator,
            test_cases,
            rng,
            history: Vec::new(),
        }
    }

    /// Establish baseline fitness from the seed genome
    pub fn establish_baseline(&mut self) -> Option<f64> {
        if let Some(genome) = self.population.first() {
            if let Some(fitness) = self.validator.fitness(genome, &self.test_cases) {
                self.baseline_fitness = fitness;
                return Some(fitness);
            }
        }
        None
    }

    /// Run one generation of evolution
    pub fn evolve_generation(&mut self) -> GenerationResult {
        self.generation += 1;

        // 1. Evaluate fitness of all genomes
        self.evaluate_population();

        // 2. Clone valid genomes sorted by fitness (lower is better)
        // We clone to avoid borrow checker issues with tournament selection
        let mut valid_genomes: Vec<Genome> = self
            .population
            .iter()
            .filter(|g| g.fitness.is_some())
            .cloned()
            .collect();
        valid_genomes.sort_by(|a, b| a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap());

        // 3. Update best ever
        if let Some(best) = valid_genomes.first() {
            if self.best_ever.is_none()
                || best.fitness.unwrap() < self.best_ever.as_ref().unwrap().fitness.unwrap()
            {
                self.best_ever = Some(best.clone());
            }
        }

        // 4. Calculate statistics
        let valid_count = valid_genomes.len();
        let (best_fitness, avg_fitness) = if valid_count > 0 {
            let best = valid_genomes.first().unwrap().fitness.unwrap();
            let sum: f64 = valid_genomes.iter().map(|g| g.fitness.unwrap()).sum();
            (best, sum / valid_count as f64)
        } else {
            (f64::MAX, f64::MAX)
        };

        let speedup = if best_fitness > 0.0 {
            self.baseline_fitness / best_fitness
        } else {
            1.0
        };

        // 5. Create next generation
        let mut next_population = Vec::with_capacity(self.config.population_size);

        // Elitism: keep best genomes unchanged
        for elite in valid_genomes.iter().take(self.config.elite_count) {
            next_population.push(elite.clone());
        }

        // Fill rest with offspring
        while next_population.len() < self.config.population_size {
            // Tournament selection for parents (using indices to avoid borrow issues)
            let parent1_idx = self.tournament_select_idx(&valid_genomes);
            let parent2_idx = self.tournament_select_idx(&valid_genomes);

            let parent1 = &valid_genomes[parent1_idx];
            let parent2 = &valid_genomes[parent2_idx];

            // Crossover
            let mut child = if self.rng.gen::<f64>() < self.config.crossover_rate {
                self.mutator.crossover(parent1, parent2)
            } else {
                parent1.clone()
            };

            // Mutation
            self.mutator.mutate(&mut child);
            child.fitness = None; // Reset fitness for re-evaluation
            child.generation = self.generation;

            next_population.push(child);
        }

        self.population = next_population;

        let result = GenerationResult {
            generation: self.generation,
            best_fitness,
            avg_fitness,
            valid_count,
            speedup_vs_baseline: speedup,
        };

        self.history.push(result.clone());
        result
    }

    /// Evaluate fitness of entire population
    fn evaluate_population(&mut self) {
        for genome in &mut self.population {
            if genome.fitness.is_none() {
                genome.fitness = self.validator.fitness(genome, &self.test_cases);
            }
        }
    }

    /// Tournament selection: returns index of best from random subset
    fn tournament_select_idx(&mut self, candidates: &[Genome]) -> usize {
        if candidates.is_empty() {
            panic!("No valid candidates for selection");
        }

        let mut best_idx = 0;
        let mut best_fitness = f64::MAX;

        for _ in 0..self.config.tournament_size.min(candidates.len()) {
            let idx = self.rng.gen_range(0..candidates.len());
            let candidate = &candidates[idx];

            if let Some(fitness) = candidate.fitness {
                if fitness < best_fitness {
                    best_fitness = fitness;
                    best_idx = idx;
                }
            }
        }

        best_idx
    }

    /// Run evolution until target speedup or max generations
    pub fn run(&mut self, max_generations: u32, target_speedup: Option<f64>) -> EvolutionResult {
        // Establish baseline
        self.establish_baseline();

        // Mutate initial population (except first which is baseline)
        for genome in self.population.iter_mut().skip(1) {
            for _ in 0..3 {
                self.mutator.mutate(genome);
            }
        }

        // Evolution loop
        for _ in 0..max_generations {
            let result = self.evolve_generation();

            // Check if target achieved
            if let Some(target) = target_speedup {
                if result.speedup_vs_baseline >= target {
                    break;
                }
            }
        }

        let best_genome = self
            .best_ever
            .clone()
            .unwrap_or_else(|| self.population.first().cloned().unwrap());

        let final_speedup = if let Some(fitness) = best_genome.fitness {
            self.baseline_fitness / fitness
        } else {
            1.0
        };

        EvolutionResult {
            best_genome,
            generations_run: self.generation,
            final_speedup,
            history: self.history.clone(),
        }
    }

    /// Get current generation number
    pub fn current_generation(&self) -> u32 {
        self.generation
    }

    /// Get the best genome found so far
    pub fn best_genome(&self) -> Option<&Genome> {
        self.best_ever.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Instruction, Opcode, Operand};

    fn create_test_function() -> Function {
        Function {
            name: "test".to_string(),
            args: vec!["x".to_string()],
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
        }
    }

    #[test]
    fn test_evolution_config_default() {
        let config = EvolutionConfig::default();
        assert_eq!(config.population_size, 50);
        assert_eq!(config.mutation_rate, 0.3);
    }

    #[test]
    fn test_engine_creation() {
        let func = create_test_function();
        let test_cases = vec![TestCase::new(0, 1), TestCase::new(10, 11)];
        let config = EvolutionConfig {
            population_size: 10,
            ..Default::default()
        };

        let engine = EvolutionEngine::new(&func, test_cases, config);
        assert_eq!(engine.population.len(), 10);
        assert_eq!(engine.current_generation(), 0);
    }
}
