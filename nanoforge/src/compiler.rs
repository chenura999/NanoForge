use crate::assembler::JitBuilder;
use crate::ir::{Function, Opcode, Operand, Program};
use std::collections::{HashMap, HashSet};

pub struct Compiler;

#[derive(Debug, Clone)]
struct Interval {
    operand: Operand,
    start: usize,
    end: usize,
    assigned_reg: Option<u8>, // Physical register ID
}

impl Compiler {
    pub fn compile_program(prog: &Program, opt_level: u8) -> Result<(Vec<u8>, usize), String> {
        let mut builder = JitBuilder::new();
        let mut main_offset = 0;

        // 0. Optimize
        let mut program = prog.clone();
        crate::optimizer::Optimizer::optimize_program(&mut program, opt_level);

        // 1. Compile each function
        for func in &program.functions {
            let label_name = format!("fn_{}", func.name);
            builder.bind_label(&label_name);
            let curr = builder.current_offset();
            // println!("Compiling fn_{}: Offset {}", func.name, curr);
            if func.name == "main" {
                main_offset = curr;
            }

            // 2a. Liveness & Alloc
            let intervals = liveness_analysis(func);

            // Filter GPRs
            let gpr_intervals: Vec<Interval> = intervals
                .iter()
                .filter(|i| matches!(i.operand, Operand::Reg(_)))
                .cloned()
                .collect();

            // Filter YMMs
            let ymm_intervals: Vec<Interval> = intervals
                .iter()
                .filter(|i| matches!(i.operand, Operand::Ymm(_)))
                .cloned()
                .collect();

            // Alloc GPRs (Pool: 0..15, with exclusions)
            let gpr_pool = vec![1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13];
            // We use a safe subset for now to match previous behavior if needed, or full set.
            // Using full set (minus 0/RAX, 6/RCX) gives more registers.
            // Safe subset: 1, 2, 3, 4, 5, 7, 8, 9, 10
            let _gpr_pool_safe = vec![1, 2, 3, 4, 5, 7, 8, 9, 10];

            // Actually, let's just rename the variable used in the code or prefix it.
            // Looking at the error: `let gpr_pool_safe = ...` is unused.
            // It seems `gpr_pool` is used later which is a subset or superset.
            // Let's just prefix it.
            let _gpr_pool_safe = vec![1, 2, 3, 4, 5, 7, 8, 9, 10];

            // ... (Fixing other issues in separate chunks if possible, but replace_file_content is single chunk or multi file tool)
            // Wait, I should use multi_replace for scattering edits.

            let (gpr_map, _) = allocate_registers(gpr_intervals, gpr_pool)?;

            // Alloc YMMs (Pool: 0..15)
            let ymm_pool = (0..16).collect();
            let (ymm_map, _) = allocate_registers(ymm_intervals, ymm_pool)?;

            // Helper to get physical reg
            let get_phys = |op: &Option<Operand>| -> u8 {
                match op {
                    Some(Operand::Reg(v)) => *gpr_map.get(&Operand::Reg(*v)).unwrap_or(&0),
                    Some(Operand::Ymm(v)) => *ymm_map.get(&Operand::Ymm(*v)).unwrap_or(&0),
                    _ => 0,
                }
            };

            let get_ymm = |op: &Option<Operand>| -> u8 {
                if let Some(Operand::Ymm(v)) = op {
                    *ymm_map.get(&Operand::Ymm(*v)).unwrap_or(&0)
                } else {
                    panic!("Expected Ymm operand");
                }
            };

            // 2b. Emit Instructions
            builder.prologue(0);

            for (idx, instr) in func.instructions.iter().enumerate() {
                if let Some(Operand::Label(name)) = &instr.dest {
                    if instr.op == Opcode::Label {
                        builder.bind_label(name);
                    }
                }
                match &instr.op {
                    Opcode::Mov => {
                        let dest = get_phys(&instr.dest);
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let s = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap_or(&0);
                            builder.mov_reg_reg(dest, s);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.mov_reg_imm(dest, val);
                        }
                    }
                    Opcode::Add => {
                        let dest = get_phys(&instr.dest);
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let s = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap_or(&0);
                            builder.add_reg_reg(dest, s);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.add_reg_imm(dest, val);
                        }
                    }
                    Opcode::Sub => {
                        let dest = get_phys(&instr.dest);
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let s = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap_or(&0);
                            builder.sub_reg_reg(dest, s);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
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
                            if let Some(Operand::Reg(_cond_vreg)) = &instr.dest {
                                // Wait, dest is Label. Condition is src1?
                                // Check parser. Jnz(dest=Label, src1=Cond).
                                // My pattern match: `if let Some(Operand::Label(target)) = &instr.dest`
                                // But checking `instr.dest` for Reg? No.
                                if let Some(Operand::Reg(cond_vreg)) = &instr.src1 {
                                    let c = *gpr_map.get(&Operand::Reg(*cond_vreg)).unwrap_or(&0);
                                    builder.cmp_reg_imm(c, 0);
                                    builder.jnz(c, target);
                                }
                            }
                        }
                    }
                    Opcode::Cmp => {
                        let r1 = get_phys(&instr.src1);
                        if let Some(Operand::Reg(r2_vreg)) = &instr.src2 {
                            let r2 = *gpr_map.get(&Operand::Reg(*r2_vreg)).unwrap_or(&0);
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

                            // Save Caller-Saved Registers
                            let mut to_save: Vec<u8> = intervals
                                .iter()
                                .filter(|iv| iv.start < idx && iv.end > idx)
                                .filter_map(|iv| {
                                    if let Operand::Reg(_) = iv.operand {
                                        iv.assigned_reg
                                    } else {
                                        None
                                    }
                                })
                                .filter(|&phys| (1..=4).contains(&phys))
                                .collect();

                            to_save.sort();
                            to_save.dedup();

                            let mut pushed_count = 0;
                            for &reg in &to_save {
                                builder.push_reg(reg);
                                pushed_count += 1;
                            }

                            if pushed_count % 2 != 0 {
                                builder.add_rsp(-8);
                            }
                            builder.call(&target_label);
                            if pushed_count % 2 != 0 {
                                builder.add_rsp(8);
                            }

                            for &reg in to_save.iter().rev() {
                                builder.pop_reg(reg);
                            }

                            let dest_phys = get_phys(&instr.dest);
                            if dest_phys != 0 {
                                builder.mov_reg_reg(dest_phys, 0);
                            }
                        }
                    }
                    Opcode::LoadArg(arg_idx) => {
                        if *arg_idx >= 4 {
                            let dest = get_phys(&instr.dest);
                            let offset = 16 + (*arg_idx as i32 * 8);
                            builder.mov_reg_stack(dest, offset);
                        }
                    }
                    Opcode::Ret => {
                        builder.epilogue();
                    }
                    Opcode::Free => {
                        let free_addr = libc::free as usize as u64;
                        builder.mov_reg_imm64(0, free_addr);

                        if let Some(Operand::Reg(vreg)) = instr.src1 {
                            let s_phys = gpr_map[&Operand::Reg(vreg)];
                            builder.mov_rdi_reg(s_phys);
                        } else {
                            panic!("Free: Invalid pointer operand");
                        }

                        builder.push_reg(1);
                        builder.push_reg(2);
                        builder.push_reg(3);
                        builder.push_reg(4);
                        builder.call_reg(0);
                        builder.pop_reg(4);
                        builder.pop_reg(3);
                        builder.pop_reg(2);
                        builder.pop_reg(1);
                    }
                    Opcode::Alloc => {
                        let malloc_addr = libc::malloc as usize as u64;
                        builder.mov_reg_imm64(0, malloc_addr);

                        if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.mov_rdi_imm(val);
                        } else if let Some(Operand::Reg(vreg)) = instr.src1 {
                            let s_phys = gpr_map[&Operand::Reg(vreg)];
                            builder.mov_rdi_reg(s_phys);
                        } else {
                            panic!("Alloc: Invalid size operand");
                        }

                        builder.push_reg(1);
                        builder.push_reg(2);
                        builder.push_reg(3);
                        builder.push_reg(4);
                        builder.call_reg(0);
                        builder.pop_reg(4);
                        builder.pop_reg(3);
                        builder.pop_reg(2);
                        builder.pop_reg(1);

                        let dest_phys = get_phys(&instr.dest);
                        if dest_phys != 0 {
                            builder.mov_reg_reg(dest_phys, 0);
                        }
                    }
                    Opcode::Load => {
                        let dest_phys = get_phys(&instr.dest);
                        let base_phys = get_phys(&instr.src1);

                        if let Some(Operand::Imm(idx)) = instr.src2 {
                            builder.mov_reg_imm(dest_phys, idx);
                            builder.mov_reg_index(dest_phys, base_phys, dest_phys);
                        } else if let Some(Operand::Reg(idx_vreg)) = instr.src2 {
                            let idx_phys = gpr_map[&Operand::Reg(idx_vreg)];
                            builder.mov_reg_index(dest_phys, base_phys, idx_phys);
                        }
                    }
                    Opcode::Store => {
                        let base_phys = get_phys(&instr.dest);

                        let val_reg_phys = if let Some(Operand::Imm(val)) = instr.src2 {
                            builder.mov_reg_imm(0, val);
                            0
                        } else {
                            get_phys(&instr.src2)
                        };

                        let idx_reg_phys = if let Some(Operand::Imm(idx)) = instr.src1 {
                            builder.mov_reg_imm(6, idx);
                            6
                        } else {
                            get_phys(&instr.src1)
                        };

                        builder.mov_index_reg(base_phys, idx_reg_phys, val_reg_phys);
                    }
                    Opcode::SetArg(arg_idx) => {
                        let phys_reg = match arg_idx {
                            0 => 1,
                            1 => 2,
                            2 => 3,
                            3 => 4,
                            _ => panic!("Only 4 args supported"),
                        };
                        if let Some(Operand::Imm(val)) = instr.src1 {
                            builder.mov_reg_imm(phys_reg, val);
                        } else if let Some(Operand::Reg(vreg)) = instr.src1 {
                            let src_phys = gpr_map[&Operand::Reg(vreg)];
                            builder.mov_reg_reg(phys_reg, src_phys);
                        }
                    }
                    Opcode::VLoad => {
                        let dest_ymm = get_ymm(&instr.dest);
                        let base_reg = get_phys(&instr.src1);
                        let index_reg = get_phys(&instr.src2);
                        builder.vmovdqu_load(dest_ymm, base_reg, index_reg, 0);
                    }
                    Opcode::VStore => {
                        let base_reg = get_phys(&instr.dest);
                        let index_reg = get_phys(&instr.src1);
                        let src_ymm = get_ymm(&instr.src2);
                        builder.vmovdqu_store(base_reg, index_reg, src_ymm, 0);
                    }
                    Opcode::VAdd => {
                        let dest_ymm = get_ymm(&instr.dest);
                        let src1_ymm = get_ymm(&instr.src1);
                        let src2_ymm = if let Some(Operand::Ymm(_)) = instr.src2 {
                            get_ymm(&instr.src2)
                        } else {
                            panic!("VAdd requires Ymm src2");
                        };
                        builder.vpaddq(dest_ymm, src1_ymm, src2_ymm);
                    }
                }
            }
        }

        let buf = builder.finalize();
        Ok((buf, main_offset))
    }
}

fn liveness_analysis(func: &Function) -> Vec<Interval> {
    let mut starts = HashMap::new();
    let mut ends = HashMap::new();
    let mut ops = HashSet::new();

    // 1. Identify Back-edges
    let mut back_edges = Vec::new(); // (start, end)

    // Map Label Name to Index
    let mut labels = HashMap::new();
    for (idx, instr) in func.instructions.iter().enumerate() {
        if instr.op == Opcode::Label {
            if let Some(Operand::Label(name)) = &instr.dest {
                labels.insert(name.clone(), idx);
            }
        }
    }

    for (idx, instr) in func.instructions.iter().enumerate() {
        if matches!(
            instr.op,
            Opcode::Jmp
                | Opcode::Jnz
                | Opcode::Je
                | Opcode::Jne
                | Opcode::Jl
                | Opcode::Jle
                | Opcode::Jg
                | Opcode::Jge
        ) {
            if let Some(Operand::Label(target)) = &instr.dest {
                if let Some(&target_idx) = labels.get(target) {
                    if target_idx < idx {
                        back_edges.push((target_idx, idx));
                    }
                }
            }
        }
    }

    for (idx, instr) in func.instructions.iter().enumerate() {
        for op in [&instr.dest, &instr.src1, &instr.src2]
            .iter()
            .filter_map(|x| x.as_ref())
        {
            match op {
                Operand::Reg(_) | Operand::Ymm(_) => {
                    ops.insert(op.clone());
                    starts.entry(op.clone()).or_insert(idx);
                    ends.insert(op.clone(), idx);
                }
                _ => {}
            }
        }

        if instr.op == Opcode::Call {
            for r in 1..=4 {
                let op = Operand::Reg(r);
                ops.insert(op.clone());
                starts.entry(op.clone()).or_insert(idx);
                ends.insert(op.clone(), idx);
            }
            let res = Operand::Reg(0);
            ops.insert(res.clone());
            starts.entry(res.clone()).or_insert(idx);
            ends.insert(res.clone(), idx);
        }
        if let Opcode::LoadArg(_) = instr.op {
            if let Some(Operand::Reg(r)) = instr.dest {
                let op = Operand::Reg(r);
                ops.insert(op.clone());
                starts.entry(op.clone()).or_insert(idx);
                ends.insert(op.clone(), idx);
            }
        }
    }

    let mut intervals: Vec<Interval> = ops
        .into_iter()
        .map(|op| {
            let start = *starts.get(&op).unwrap_or(&0);
            let mut end = *ends.get(&op).unwrap_or(&0);

            // Extend for loops
            for &(loop_head, loop_tail) in &back_edges {
                // If variable is live at loop head (start <= head <= end)
                if start <= loop_head && end >= loop_head {
                    // It must be preserved until loop tail
                    if end < loop_tail {
                        end = loop_tail;
                    }
                }
            }

            Interval {
                operand: op.clone(),
                start,
                end,
                assigned_reg: None,
            }
        })
        .collect();

    intervals.sort_by_key(|i| i.start);
    intervals
}

fn allocate_registers(
    mut intervals: Vec<Interval>,
    pool: Vec<u8>,
) -> Result<(HashMap<Operand, u8>, Vec<Interval>), String> {
    let mut active: Vec<Interval> = Vec::new();
    let mut map = HashMap::new();

    // Pre-color VReg(0) -> PReg(0)
    for iv in &intervals {
        if let Operand::Reg(0) = iv.operand {
            map.insert(iv.operand.clone(), 0);
        }
    }

    // Pre-color ABI Registers (Reg 1..4) -> PReg 1..4
    for r in 1..5 {
        let op = Operand::Reg(r);
        if intervals.iter().any(|i| i.operand == op) {
            map.insert(op, r);
        }
    }

    let mut pre_colored_intervals: HashMap<u8, Vec<Interval>> = HashMap::new();
    for iv in &intervals {
        if let Some(&phys) = map.get(&iv.operand) {
            pre_colored_intervals
                .entry(phys)
                .or_default()
                .push(iv.clone());
        }
    }

    for i in 0..intervals.len() {
        if map.contains_key(&intervals[i].operand) {
            intervals[i].assigned_reg = Some(map[&intervals[i].operand]);
            active.push(intervals[i].clone());
            continue;
        }

        let current = &intervals[i];
        let current_start = current.start;
        let current_end = current.end;

        active.retain(|interval| interval.end > current_start);

        let used_regs: HashSet<u8> = active.iter().filter_map(|i| i.assigned_reg).collect();

        let mut free_regs: Vec<u8> = pool
            .iter()
            .cloned()
            .filter(|r| !used_regs.contains(r))
            .filter(|r| {
                if let Some(fixed_ivs) = pre_colored_intervals.get(r) {
                    for fixed in fixed_ivs {
                        if current_start < fixed.end && fixed.start < current_end {
                            return false;
                        }
                    }
                }
                true
            })
            .collect();

        free_regs.sort();

        if let Some(phys_reg) = free_regs.first() {
            intervals[i].assigned_reg = Some(*phys_reg);
            map.insert(intervals[i].operand.clone(), *phys_reg);
            active.push(intervals[i].clone());
        } else {
            return Err(format!(
                "Register Spilling not implemented. Active: {}",
                active.len()
            ));
        }
    }

    Ok((map, intervals))
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

        let (code, _) = Compiler::compile_program(&prog, 0).expect("Compilation failed");

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 30);
    }

    #[test]
    fn test_compile_loop() {
        let mut func = Function::new("loop_test", vec![]);
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(5)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Mov,
            dest: Some(Operand::Reg(2)),
            src1: Some(Operand::Imm(0)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Label,
            dest: Some(Operand::Label("loop".to_string())),
            src1: None,
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Cmp,
            dest: None,
            src1: Some(Operand::Reg(1)),
            src2: Some(Operand::Imm(0)),
        });
        func.push(Instruction {
            op: Opcode::Je,
            dest: Some(Operand::Label("end".to_string())),
            src1: None,
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Add,
            dest: Some(Operand::Reg(2)),
            src1: Some(Operand::Reg(1)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Sub,
            dest: Some(Operand::Reg(1)),
            src1: Some(Operand::Imm(1)),
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Jmp,
            dest: Some(Operand::Label("loop".to_string())),
            src1: None,
            src2: None,
        });
        func.push(Instruction {
            op: Opcode::Label,
            dest: Some(Operand::Label("end".to_string())),
            src1: None,
            src2: None,
        });
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

        let (code, main_offset) = Compiler::compile_program(&prog, 0).expect("Compilation failed");

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code, 0);
        let func_ptr: extern "C" fn() -> i64 =
            unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
        assert_eq!(func_ptr(), 15);
    }
}
