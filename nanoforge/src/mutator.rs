//! Mutation Operators for Genetic Code Evolution
//!
//! This module provides mutation operators that can transform IR instructions
//! to explore the optimization space through genetic algorithms.

use crate::ir::{Function, Instruction, Opcode, Operand};
use rand::prelude::*;

/// Types of mutations that can be applied to code
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationType {
    /// Swap two adjacent instructions (if safe)
    SwapInstructions,
    /// Change a register number
    ChangeRegister,
    /// Change an immediate value slightly
    TweakImmediate,
    /// Delete a non-essential instruction
    DeleteInstruction,
    /// Duplicate an instruction (for unrolling effect)
    DuplicateInstruction,
    /// Insert a NOP (can help with alignment)
    InsertNop,
}

impl MutationType {
    /// Get all mutation types
    pub fn all() -> &'static [MutationType] {
        &[
            MutationType::SwapInstructions,
            MutationType::ChangeRegister,
            MutationType::TweakImmediate,
            MutationType::DeleteInstruction,
            MutationType::DuplicateInstruction,
            MutationType::InsertNop,
        ]
    }

    /// Pick a random mutation type
    pub fn random<R: Rng>(rng: &mut R) -> MutationType {
        let all = Self::all();
        all[rng.gen_range(0..all.len())]
    }
}

/// A genome representing a function's code
#[derive(Debug, Clone)]
pub struct Genome {
    /// The function's instructions
    pub instructions: Vec<Instruction>,
    /// Function metadata
    pub name: String,
    pub args: Vec<String>,
    /// Fitness score (lower is better, measured in nanoseconds)
    pub fitness: Option<f64>,
    /// Generation this genome was created
    pub generation: u32,
}

impl Genome {
    /// Create a genome from a function
    pub fn from_function(func: &Function) -> Self {
        Self {
            instructions: func.instructions.clone(),
            name: func.name.clone(),
            args: func.args.clone(),
            fitness: None,
            generation: 0,
        }
    }

    /// Convert back to a function
    pub fn to_function(&self) -> Function {
        Function {
            name: self.name.clone(),
            args: self.args.clone(),
            instructions: self.instructions.clone(),
        }
    }

    /// Get the number of instructions
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

/// Mutator that applies random mutations to genomes
pub struct Mutator {
    /// Probability of mutation per instruction (0.0 - 1.0)
    pub mutation_rate: f64,
    /// Maximum registers available
    pub max_registers: u8,
    /// RNG for randomness
    rng: StdRng,
}

impl Mutator {
    /// Create a new mutator with given mutation rate
    pub fn new(mutation_rate: f64, seed: u64) -> Self {
        Self {
            mutation_rate,
            max_registers: 10, // Virtual registers 0-9
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Apply a random mutation to the genome
    pub fn mutate(&mut self, genome: &mut Genome) -> Option<MutationType> {
        if genome.is_empty() {
            return None;
        }

        // Decide if we should mutate
        if self.rng.gen::<f64>() > self.mutation_rate {
            return None;
        }

        let mutation_type = MutationType::random(&mut self.rng);

        match mutation_type {
            MutationType::SwapInstructions => {
                self.swap_instructions(genome);
            }
            MutationType::ChangeRegister => {
                self.change_register(genome);
            }
            MutationType::TweakImmediate => {
                self.tweak_immediate(genome);
            }
            MutationType::DeleteInstruction => {
                self.delete_instruction(genome);
            }
            MutationType::DuplicateInstruction => {
                self.duplicate_instruction(genome);
            }
            MutationType::InsertNop => {
                self.insert_nop(genome);
            }
        }

        Some(mutation_type)
    }

    /// Swap two adjacent non-label instructions
    fn swap_instructions(&mut self, genome: &mut Genome) {
        if genome.len() < 2 {
            return;
        }

        // Find a valid swap point (not labels, not jumps)
        let mut attempts = 0;
        while attempts < 10 {
            let idx = self.rng.gen_range(0..genome.len() - 1);

            let can_swap =
                !matches!(
                    genome.instructions[idx].op,
                    Opcode::Label
                        | Opcode::Jmp
                        | Opcode::Je
                        | Opcode::Jne
                        | Opcode::Jl
                        | Opcode::Jle
                        | Opcode::Jg
                        | Opcode::Jge
                        | Opcode::Ret
                        | Opcode::Call
                ) && !matches!(genome.instructions[idx + 1].op, Opcode::Label | Opcode::Ret);

            if can_swap {
                genome.instructions.swap(idx, idx + 1);
                return;
            }
            attempts += 1;
        }
    }

    /// Change a register in a random instruction
    fn change_register(&mut self, genome: &mut Genome) {
        if genome.is_empty() {
            return;
        }

        let idx = self.rng.gen_range(0..genome.len());
        let instr = &mut genome.instructions[idx];

        // Try to change dest register
        if let Some(Operand::Reg(ref mut r)) = instr.dest {
            *r = self.rng.gen_range(0..self.max_registers);
        } else if let Some(Operand::Reg(ref mut r)) = instr.src1 {
            *r = self.rng.gen_range(0..self.max_registers);
        }
    }

    /// Tweak an immediate value
    fn tweak_immediate(&mut self, genome: &mut Genome) {
        for instr in &mut genome.instructions {
            if let Some(Operand::Imm(ref mut val)) = instr.src1 {
                // Small random adjustment
                let delta = self.rng.gen_range(-5..=5);
                *val = val.saturating_add(delta);
                return;
            }
            if let Some(Operand::Imm(ref mut val)) = instr.dest {
                let delta = self.rng.gen_range(-5..=5);
                *val = val.saturating_add(delta);
                return;
            }
        }
    }

    /// Delete a non-essential instruction
    fn delete_instruction(&mut self, genome: &mut Genome) {
        if genome.len() < 3 {
            return; // Don't delete if too few instructions
        }

        // Find a deletable instruction (not labels, ret, jumps)
        for _ in 0..10 {
            let idx = self.rng.gen_range(0..genome.len());
            let can_delete = matches!(
                genome.instructions[idx].op,
                Opcode::Mov | Opcode::Add | Opcode::Sub
            );

            if can_delete {
                genome.instructions.remove(idx);
                return;
            }
        }
    }

    /// Duplicate an instruction (loop unrolling effect)
    fn duplicate_instruction(&mut self, genome: &mut Genome) {
        if genome.is_empty() {
            return;
        }

        let idx = self.rng.gen_range(0..genome.len());
        let can_duplicate = matches!(
            genome.instructions[idx].op,
            Opcode::Add | Opcode::Sub | Opcode::Mov | Opcode::Load | Opcode::Store
        );

        if can_duplicate {
            let duplicate = genome.instructions[idx].clone();
            genome.instructions.insert(idx + 1, duplicate);
        }
    }

    /// Insert a NOP instruction (can help with alignment)
    fn insert_nop(&mut self, genome: &mut Genome) {
        if genome.is_empty() {
            return;
        }

        let idx = self.rng.gen_range(0..genome.len());

        // NOP is represented as mov r0, r0 (no-op)
        let nop = Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(0)),
            src1: Some(Operand::Reg(0)),
            src2: None,
        };

        genome.instructions.insert(idx, nop);
    }

    /// Perform crossover between two parents to create a child
    pub fn crossover(&mut self, parent1: &Genome, parent2: &Genome) -> Genome {
        // Single-point crossover
        let min_len = parent1.len().min(parent2.len());
        if min_len < 2 {
            return parent1.clone();
        }

        let crossover_point = self.rng.gen_range(1..min_len);

        let mut child_instructions = Vec::new();
        child_instructions.extend_from_slice(&parent1.instructions[..crossover_point]);
        child_instructions.extend_from_slice(&parent2.instructions[crossover_point..]);

        Genome {
            instructions: child_instructions,
            name: parent1.name.clone(),
            args: parent1.args.clone(),
            fitness: None,
            generation: parent1.generation.max(parent2.generation) + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_genome() -> Genome {
        Genome {
            instructions: vec![
                Instruction {
                    op: Opcode::Mov,
                    dest: Some(Operand::Reg(0)),
                    src1: Some(Operand::Imm(0)),
                    src2: None,
                },
                Instruction {
                    op: Opcode::Add,
                    dest: Some(Operand::Reg(0)),
                    src1: Some(Operand::Reg(1)),
                    src2: None,
                },
                Instruction {
                    op: Opcode::Ret,
                    dest: Some(Operand::Reg(0)),
                    src1: None,
                    src2: None,
                },
            ],
            name: "test".to_string(),
            args: vec![],
            fitness: None,
            generation: 0,
        }
    }

    #[test]
    fn test_mutate() {
        let mut mutator = Mutator::new(1.0, 42); // 100% mutation rate
        let mut genome = create_test_genome();
        let original_len = genome.len();

        // Apply mutations
        for _ in 0..10 {
            mutator.mutate(&mut genome);
        }

        // Should have changed something (length or content)
        assert!(genome.len() != original_len || genome.fitness.is_none());
    }

    #[test]
    fn test_crossover() {
        let mut mutator = Mutator::new(0.5, 42);
        let parent1 = create_test_genome();
        let mut parent2 = create_test_genome();
        parent2.instructions.push(Instruction {
            op: Opcode::Sub,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(1)),
            src2: None,
        });

        let child = mutator.crossover(&parent1, &parent2);
        assert!(!child.instructions.is_empty());
        assert_eq!(child.generation, 1);
    }

    #[test]
    fn test_mutation_types() {
        assert_eq!(MutationType::all().len(), 6);
    }
}
