use crate::assembler::JitBuilder;
use crate::ir::{Function, Opcode, Operand, Program};
use std::collections::{HashMap, HashSet};

pub struct Compiler;

#[derive(Debug, Clone)]
struct Interval {
    vreg: u8,
    start: usize,
    end: usize,
    assigned_reg: Option<u8>, // Physical register ID (0..N)
}

impl Compiler {
    pub fn compile_program(prog: &Program) -> Result<(Vec<u8>, usize), String> {
        let mut builder = JitBuilder::new();
        let mut main_offset = 0;

        // 1. Compile each function
        for func in &prog.functions {
            let label_name = format!("fn_{}", func.name);
            builder.bind_label(&label_name);
            let curr = builder.current_offset();
            println!("Compiling fn_{}: Offset {}", func.name, curr);
            if func.name == "main" {
                main_offset = curr;
            }

            // 2a. Liveness & Alloc
            let intervals = liveness_analysis(func);
            let allocation_map = allocate_registers(intervals)?;

            // Create a simple map for easier lookup (u8 -> u8)
            let mut reg_map = HashMap::new();
            for (vreg, phys) in &allocation_map {
                reg_map.insert(*vreg, *phys);
            }

            // Helper to get physical reg
            // Captures reg_map
            let get_phys = |op: &Option<Operand>, map: &HashMap<u8, u8>| -> u8 {
                if let Some(Operand::Reg(vreg)) = op {
                    *map.get(vreg).unwrap_or(&0)
                } else {
                    0
                }
            };

            // 2b. Emit Instructions
            // HARDCODED stack size 0 for now (Allocator handles spilling via Registers, recursion uses Stack via Push/Pop)
            builder.prologue(0);

            for (_idx, instr) in func.instructions.iter().enumerate() {
                // println!("Processing Op: {:?}", instr.op);
                if let Some(label) = &instr.dest {
                    if let Operand::Label(name) = label {
                        if instr.op == Opcode::Label {
                            println!("BINDING LABEL: {}", name);
                            builder.bind_label(name);
                        }
                    }
                }
                match instr.op {
                    Opcode::Mov => {
                        let dest = get_phys(&instr.dest, &reg_map);
                        if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.mov_reg_imm(dest, val);
                        } else if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let s = *reg_map.get(&src_vreg).unwrap_or(&0);
                            builder.mov_reg_reg(dest, s);
                        }
                    }
                    Opcode::Add => {
                        let dest = get_phys(&instr.dest, &reg_map);
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let s = *reg_map.get(&src_vreg).unwrap_or(&0);
                            builder.add_reg_reg(dest, s);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.add_reg_imm(dest, val);
                        }
                    }
                    Opcode::Sub => {
                        let dest = get_phys(&instr.dest, &reg_map);
                        if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.sub_reg_imm(dest, val);
                        }
                    }

                    Opcode::Label => {}
                    Opcode::Jmp => {
                        if let Some(Operand::Label(target)) = &instr.dest {
                            builder.jmp(target);
                        }
                    }
                    Opcode::Jnz => {
                        if let Some(Operand::Label(target)) = &instr.dest {
                            if let Some(Operand::Reg(cond_vreg)) = &instr.dest {
                                let c = *reg_map.get(cond_vreg).unwrap_or(&0);
                                builder.cmp_reg_imm(c, 0); // Check if != 0
                                builder.jnz(c, target);
                            }
                        }
                    }
                    Opcode::Cmp => {
                        let r1 = get_phys(&instr.src1, &reg_map);
                        if let Some(Operand::Reg(r2_vreg)) = &instr.src2 {
                            let r2 = *reg_map.get(r2_vreg).unwrap_or(&0);
                            builder.cmp_reg_reg(r1, r2);
                        } else if let Some(Operand::Imm(val)) = &instr.src2 {
                            builder.cmp_reg_imm(r1, *val);
                        }
                    }
                    Opcode::Je => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.je(t);
                        }
                    }
                    Opcode::Jne => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.jne(t);
                        }
                    }
                    Opcode::Jl => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.jl(t);
                        }
                    }
                    Opcode::Jle => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.jle(t);
                        }
                    }
                    Opcode::Jg => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.jg(t);
                        }
                    }
                    Opcode::Jge => {
                        if let Some(Operand::Label(t)) = &instr.dest {
                            builder.jge(t);
                        }
                    }
                    Opcode::Call => {
                        if let Some(Operand::Label(target)) = &instr.src1 {
                            let target_label = format!("fn_{}", target);

                            builder.call(&target_label);

                            // Stack alignment note: 4 pushes (32 bytes) + Call (8) + RBP (8) ...
                            // If Stack was aligned 16 before 'push_reg',
                            // 32 bytes keeps it aligned.
                            // Call pushes 8 -> Misaligned (8).
                            // Prologue pushes RBP (16). Aligned.
                            // This looks safe.
                            let dest_phys = get_phys(&instr.dest, &reg_map);
                            if dest_phys != 0 {
                                builder.mov_reg_reg(dest_phys, 0); // Mov Dest, RAX
                            }
                        }
                    }
                    Opcode::LoadArg(idx) => {
                        // Args 0..3 passed in Reg 1..4 (R8..R11).
                        // No need to load from stack unless spilled (which Allocator handles via Reg Map?)
                        // Allocator pre-colors Reg 1..4.
                        // So if we are in Reg 1..4, VALUE IS ALREADY THERE.
                        // ONLY load from stack if idx >= 4?
                        // For now, our ABI supports 4 register args.
                        if idx >= 4 {
                            let dest = get_phys(&instr.dest, &reg_map);
                            let offset = 16 + (idx as i32 * 8);
                            builder.mov_reg_stack(dest, offset);
                        }
                    }
                    Opcode::Ret => {
                        builder.epilogue();
                    }
                }
            }
        }
        let code = builder.finalize();
        Ok((code, main_offset))
    }
}

fn liveness_analysis(func: &Function) -> Vec<Interval> {
    let mut starts = HashMap::new();
    let mut ends = HashMap::new();
    let mut vregs = HashSet::new();

    for (idx, instr) in func.instructions.iter().enumerate() {
        for op in [&instr.dest, &instr.src1, &instr.src2]
            .iter()
            .filter_map(|x| x.as_ref())
        {
            if let Operand::Reg(r) = op {
                vregs.insert(*r);
                starts.entry(*r).or_insert(idx);
                ends.insert(*r, idx); // Update end to current
            }
        }

        // Implicit Uses for Call
        if instr.op == Opcode::Call {
            // Call uses Reg(1)..Reg(4) (Args) - Reduced to allow locals in Reg 5
            for r in 1..=4 {
                vregs.insert(r);
                starts.entry(r).or_insert(idx); // Likely defined earlier, but ensure entry
                ends.insert(r, idx);
            }
            // Call defines Reg(0) (Result)
            vregs.insert(0);
            starts.entry(0).or_insert(idx);
            ends.insert(0, idx);
        }
        // Implicit definition for LoadArg
        if let Opcode::LoadArg(_) = instr.op {
            if let Some(Operand::Reg(r)) = instr.dest {
                vregs.insert(r);
                starts.entry(r).or_insert(idx);
                ends.insert(r, idx);
            }
        }
    }

    let mut intervals: Vec<Interval> = vregs
        .into_iter()
        .map(|vreg| Interval {
            vreg,
            start: *starts.get(&vreg).unwrap_or(&0),
            end: *ends.get(&vreg).unwrap_or(&0),
            assigned_reg: None,
        })
        .collect();

    // Sort by start position for Linear Scan
    intervals.sort_by_key(|i| i.start);
    intervals
}

fn allocate_registers(mut intervals: Vec<Interval>) -> Result<HashMap<u8, u8>, String> {
    // Available physical registers: 0, 1, 2 (rax, rcx, rdx / x0, x1, x2)
    // We RESERVE Physical Register 0 (RAX) for Virtual Register 0.
    // This allows the Parser to use VReg(0) as the "Return Value" holder.
    let total_regs = 6;
    let mut active: Vec<Interval> = Vec::new();
    let mut map = HashMap::new();

    // Pre-color VReg(0) -> PReg(0) if it exists
    if intervals.iter().any(|i| i.vreg == 0) {
        map.insert(0, 0);
    }

    // Pre-color ABI Registers (Reg 1..5) -> PReg 1..5
    // These represent Arguments passed in specific registers.
    // They MUST map to their physical counterparts.
    for r in 1..total_regs {
        if intervals.iter().any(|i| i.vreg == r) {
            map.insert(r, r);
        }
    }

    // Available for general allocation: 1, 2...
    // The `initial_free_regs` variable was removed as it was not used.
    // The filtering for free_regs is now done directly in the loop.

    for i in 0..intervals.len() {
        if intervals[i].vreg < total_regs {
            // Already handled (pre-colored)
            intervals[i].assigned_reg = Some(intervals[i].vreg);
            active.push(intervals[i].clone());
            continue;
        }

        let current_start = intervals[i].start;

        // Expire old intervals
        // Use <= to allow reuse at the same instruction index (last use)
        active.retain(|interval| interval.end > current_start);

        // Re-calculate free regs
        // Only consider regs 1..N
        let used_regs: HashSet<u8> = active.iter().filter_map(|i| i.assigned_reg).collect();
        let mut free_regs: Vec<u8> = (1..total_regs).filter(|r| !used_regs.contains(r)).collect();
        free_regs.sort();

        // Allocate
        if let Some(phys_reg) = free_regs.first() {
            intervals[i].assigned_reg = Some(*phys_reg);
            map.insert(intervals[i].vreg, *phys_reg);
            active.push(intervals[i].clone());
        } else {
            return Err(format!(
                "Out of registers! Active: {:?}, Current: {:?}",
                active, intervals[i]
            ));
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::CodeGenerator;
    use crate::ir::Instruction;
    use crate::jit_memory::DualMappedMemory;

    #[test]
    fn test_compile_add_imm() {
        let mut func = Function::new("test", vec![]);
        // x = 10 + 5
        // mov reg(1), 10
        // mov reg(2), 5  (opt)
        // add reg(1), reg(2) ? No, Add Reg, Imm support?
        // IR Add is Reg, Reg. Parser handles Imm -> Reg.
        // Mov Reg(1), 10
        // Mov Reg(0), Reg(1)
        // Add Reg(0), 5 ??? Parser adds imm to reg?
        // Let's replicate what parser does.
        // Mov x, 10
        // Mov y, 20
        // Add x, y
        // Ret x

        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(10)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(2)),
            src1: Some(Operand::Imm(20)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Add,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Reg(2)),
            src2: None,
        });
        // Return x (Reg 1) -> Mov Reg 0, Reg 1; Ret
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(0)),
            src1: Some(Operand::Reg(1)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Ret,
            dest: None,
            src1: None,
            src2: None,
        });

        let mut prog = Program::new();
        prog.add_function(func);

        let code = Compiler::compile_program(&prog).expect("Compilation failed");

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code.0, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 30);
    }

    #[test]
    fn test_compile_loop() {
        let mut func = Function::new("loop_test", vec![]);
        // i = 5; sum = 0
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(5)),
            src2: None,
        }); // i
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(2)),
            src1: Some(Operand::Imm(0)),
            src2: None,
        }); // sum hiding in Reg 2

        // label loop
        func.push(Instruction {
            op: Opcode::Label,
            dest: Some(Operand::Label("loop".to_string())),
            src1: None,
            src2: None,
        });

        // if i == 0 goto end
        // Cmp i, 0
        func.push(Instruction {
            op: Opcode::Cmp,
            dest: None,
            src1: Some(Operand::Reg(1)),
            src2: Some(Operand::Imm(0)),
        });
        // Je end
        func.push(Instruction {
            op: Opcode::Je,
            dest: Some(Operand::Label("end".to_string())),
            src1: None,
            src2: None,
        });

        // sum = sum + i
        func.push(Instruction {
            op: Opcode::Add,
            dest: Some(Operand::Reg(2)),
            src1: Some(Operand::Reg(1)),
            src2: None,
        });

        // i = i - 1
        func.push(Instruction {
            op: Opcode::Sub,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(1)),
            src2: None,
        });

        // goto loop
        func.push(Instruction {
            op: Opcode::Jmp,
            dest: Some(Operand::Label("loop".to_string())),
            src1: None,
            src2: None,
        });

        // label end
        func.push(Instruction {
            op: Opcode::Label,
            dest: Some(Operand::Label("end".to_string())),
            src1: None,
            src2: None,
        });

        // ret sum
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(0)),
            src1: Some(Operand::Reg(2)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Ret,
            dest: None,
            src1: None,
            src2: None,
        });

        let mut prog = Program::new();
        prog.add_function(func);

        let (code, main_offset) = Compiler::compile_program(&prog).expect("Compilation failed");

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code, 0);
        let func_ptr: extern "C" fn() -> i64 =
            unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
        assert_eq!(func_ptr(), 15);
    }
}
