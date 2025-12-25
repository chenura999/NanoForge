use crate::assembler::JitBuilder;
use crate::ir::{Function, Opcode, Operand, Program};
use std::collections::{HashMap, HashSet};

pub struct Compiler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Register(u8),
    Spill(i32), // Stack offset relative to RBP
}

#[derive(Debug, Clone)]
struct Interval {
    operand: Operand,
    start: usize,
    end: usize,
    assigned_loc: Option<Location>,
}

impl Compiler {
    pub fn compile_program(prog: &Program, opt_level: u8) -> Result<(Vec<u8>, usize), String> {
        let mut builder = JitBuilder::new();
        let mut main_offset = 0;

        let mut program = prog.clone();
        crate::optimizer::Optimizer::optimize_program(&mut program, opt_level);

        for func in &program.functions {
            let label_name = format!("fn_{}", func.name);
            let fail_label = format!("fuel_fail_{}", func.name);
            
            builder.bind_label(&label_name);
            let curr = builder.current_offset();
            if func.name == "main" {
                main_offset = curr;
            }

            let intervals = liveness_analysis(func);

            let gpr_intervals: Vec<Interval> = intervals
                .iter()
                .filter(|i| matches!(i.operand, Operand::Reg(_)))
                .cloned()
                .collect();

            let ymm_intervals: Vec<Interval> = intervals
                .iter()
                .filter(|i| matches!(i.operand, Operand::Ymm(_)))
                .cloned()
                .collect();

            let gpr_pool = vec![1, 2, 3, 4, 7, 8, 11, 12, 13]; 
            let scratch1 = 9;  // R13
            let scratch2 = 10; // R14

            let callee_saved_size = 40;

            let (gpr_map, stack_slots) = allocate_registers(gpr_intervals, gpr_pool, callee_saved_size)?;
            
            let spill_slots = stack_slots;
            let raw_stack_size = spill_slots * 8;
            
            let mut stack_size = raw_stack_size;
            if stack_size % 16 == 0 {
                stack_size += 8;
            }

            let ymm_pool = (0..16).collect();
            let (ymm_map, _) = allocate_registers(ymm_intervals, ymm_pool, 0)?;

            let get_loc = |op: &Option<Operand>| -> Location {
                match op {
                    Some(Operand::Reg(v)) => *gpr_map.get(&Operand::Reg(*v)).unwrap_or(&Location::Register(0)),
                    _ => Location::Register(0),
                }
            };

            let _get_ymm = |op: &Option<Operand>| -> u8 {
                if let Some(Operand::Ymm(v)) = op {
                    if let Some(Location::Register(r)) = ymm_map.get(&Operand::Ymm(*v)) {
                         *r
                    } else {
                        0
                    }
                } else {
                    panic!("Expected Ymm operand");
                }
            };

            builder.prologue(0); 
            
            builder.push_reg(7);
            builder.push_reg(8);
            builder.push_reg(9);
            builder.push_reg(10);
            builder.push_reg(5);
            
            if stack_size > 0 {
                builder.add_rsp(-stack_size);
            }
            
            builder.mov_reg_imm(5, 1_000_000);

            let mut label_indices = HashMap::new();
            for (i, instr) in func.instructions.iter().enumerate() {
                if let Opcode::Label = instr.op {
                    if let Some(Operand::Label(name)) = &instr.dest {
                        label_indices.insert(name.clone(), i);
                    }
                }
            }
            let mut loop_headers = HashSet::new();
            for (i, instr) in func.instructions.iter().enumerate() {
                let target_label = match instr.op {
                    Opcode::Jmp | Opcode::Jnz | Opcode::Je | Opcode::Jne | 
                    Opcode::Jl | Opcode::Jle | Opcode::Jg | Opcode::Jge => {
                        if let Some(Operand::Label(target)) = &instr.dest {
                            Some(target)
                        } else { None }
                    }
                    _ => None
                };
                if let Some(target) = target_label {
                    if let Some(&target_idx) = label_indices.get(target) {
                        if target_idx < i {
                            loop_headers.insert(target.clone());
                        }
                    }
                }
            }

            for (idx, instr) in func.instructions.iter().enumerate() {
                let load_op = |builder: &mut JitBuilder, loc: Location, scratch: u8| -> u8 {
                    match loc {
                        Location::Register(r) => r,
                        Location::Spill(offset) => {
                            builder.mov_reg_stack(scratch, offset);
                            scratch
                        }
                    }
                };

                let store_op = |builder: &mut JitBuilder, loc: Location, src_reg: u8| {
                    match loc {
                        Location::Register(r) => {
                            if r != src_reg {
                                builder.mov_reg_reg(r, src_reg);
                            }
                        }
                        Location::Spill(offset) => {
                            builder.mov_stack_reg(offset, src_reg);
                        }
                    }
                };
                
                if let Some(Operand::Label(name)) = &instr.dest {
                     if instr.op == Opcode::Label {
                        builder.bind_label(name);
                        if loop_headers.contains(name) {
                            builder.dec_reg(5); 
                            builder.jz(&fail_label);
                        }
                     }
                }

                match &instr.op {
                    Opcode::Mov => {
                        let dest_loc = get_loc(&instr.dest);
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                            let src_loc = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap();
                            match (dest_loc, src_loc) {
                                (Location::Register(d), Location::Register(s)) => builder.mov_reg_reg(d, s),
                                (Location::Register(d), Location::Spill(off)) => builder.mov_reg_stack(d, off),
                                (Location::Spill(off), Location::Register(s)) => builder.mov_stack_reg(off, s),
                                (Location::Spill(d_off), Location::Spill(s_off)) => {
                                    builder.mov_reg_stack(scratch1, s_off);
                                    builder.mov_stack_reg(d_off, scratch1);
                                }
                            }
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                            match dest_loc {
                                Location::Register(d) => builder.mov_reg_imm(d, val),
                                Location::Spill(off) => {
                                    builder.mov_reg_imm(scratch1, val);
                                    builder.mov_stack_reg(off, scratch1);
                                }
                            }
                        }
                    }
                    Opcode::Add => {
                        let dest_loc = get_loc(&instr.dest);
                        let d_reg = load_op(&mut builder, dest_loc, scratch1);
                        
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap();
                             let s_reg = load_op(&mut builder, src_loc, scratch2);
                             builder.add_reg_reg(d_reg, s_reg);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                             builder.add_reg_imm(d_reg, val);
                        }
                        
                        if let Location::Spill(off) = dest_loc {
                            builder.mov_stack_reg(off, d_reg);
                        }
                    }
                     Opcode::Sub => {
                        let dest_loc = get_loc(&instr.dest);
                        let d_reg = load_op(&mut builder, dest_loc, scratch1);
                        
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap();
                             let s_reg = load_op(&mut builder, src_loc, scratch2);
                             builder.sub_reg_reg(d_reg, s_reg);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                             builder.sub_reg_imm(d_reg, val);
                        }
                        if let Location::Spill(off) = dest_loc {
                            builder.mov_stack_reg(off, d_reg);
                        }
                    }
                    Opcode::Mul => {
                        let dest_loc = get_loc(&instr.dest);
                        let d_reg = load_op(&mut builder, dest_loc, scratch1);
                        
                        if let Some(Operand::Reg(src_vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(src_vreg)).unwrap();
                             let s_reg = load_op(&mut builder, src_loc, scratch2);
                             builder.imul_reg_reg(d_reg, s_reg);
                        } else if let Some(Operand::Imm(val)) = instr.src1 {
                             builder.imul_reg_imm(d_reg, val);
                        }
                        if let Location::Spill(off) = dest_loc {
                            builder.mov_stack_reg(off, d_reg);
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
                            if let Some(Operand::Reg(cond_vreg)) = &instr.src1 {
                                 let cond_loc = *gpr_map.get(&Operand::Reg(*cond_vreg)).unwrap();
                                 let c_reg = load_op(&mut builder, cond_loc, scratch1);
                                 builder.cmp_reg_imm(c_reg, 0);
                                 builder.jnz(c_reg, target);
                            }
                        }
                    }
                     Opcode::Cmp => {
                        let r1_loc = get_loc(&instr.src1);
                        let r1 = load_op(&mut builder, r1_loc, scratch1);
                        
                        if let Some(Operand::Reg(r2_vreg)) = &instr.src2 {
                            let r2_loc = *gpr_map.get(&Operand::Reg(*r2_vreg)).unwrap();
                            let r2 = load_op(&mut builder, r2_loc, scratch2);
                            builder.cmp_reg_reg(r1, r2);
                        } else if let Some(Operand::Imm(val)) = &instr.src2 {
                            builder.cmp_reg_imm(r1, *val);
                        }
                    }
                    Opcode::Je => { if let Some(Operand::Label(t)) = &instr.dest { builder.je(t); } }
                    Opcode::Jne => { if let Some(Operand::Label(t)) = &instr.dest { builder.jne(t); } }
                    Opcode::Jl => { if let Some(Operand::Label(t)) = &instr.dest { builder.jl(t); } }
                    Opcode::Jle => { if let Some(Operand::Label(t)) = &instr.dest { builder.jle(t); } }
                    Opcode::Jg => { if let Some(Operand::Label(t)) = &instr.dest { builder.jg(t); } }
                    Opcode::Jge => { if let Some(Operand::Label(t)) = &instr.dest { builder.jge(t); } }

                    Opcode::LoadArg(arg_idx) => {
                         let dest_loc = get_loc(&instr.dest);
                         let src_phys = match arg_idx {
                                 0 => 11,
                                 1 => 12,
                                 2 => 13,
                                 3 => 6,
                                 _ => panic!("Max 4 args"),
                         };
                         store_op(&mut builder, dest_loc, src_phys);
                    }
                    Opcode::SetArg(arg_idx) => {
                         let dest_phys = match arg_idx {
                                 0 => 11,
                                 1 => 12,
                                 2 => 13,
                                 3 => 6,
                                 _ => panic!("Max 4 args"),
                         };
                         if let Some(Operand::Imm(val)) = instr.src1 {
                             builder.mov_reg_imm(dest_phys, val);
                         } else if let Some(Operand::Reg(vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(vreg)).unwrap();
                             let s = load_op(&mut builder, src_loc, scratch1);
                             if s != dest_phys {
                                builder.mov_reg_reg(dest_phys, s);
                             }
                         }
                    }
                    Opcode::Call => {
                         if let Some(Operand::Label(target)) = &instr.src1 {
                            let target_label = format!("fn_{}", target);
                            
                            let mut to_save: Vec<u8> = intervals
                                .iter()
                                .filter(|iv| iv.start < idx && iv.end > idx)
                                .filter_map(|iv| {
                                     match iv.assigned_loc {
                                         Some(Location::Register(r)) => Some(r),
                                         _ => None
                                     }
                                })
                                .filter(|&r| is_caller_saved(r)) 
                                .collect();
                            
                            to_save.sort();
                            to_save.dedup();

                            let mut pushed_count = 0;
                            for &reg in &to_save {
                                builder.push_reg(reg);
                                pushed_count += 1;
                            }
                            if pushed_count % 2 != 0 { builder.add_rsp(-8); }
                            
                            builder.call(&target_label);
                            
                            if pushed_count % 2 != 0 { builder.add_rsp(8); }
                             for &reg in to_save.iter().rev() {
                                builder.pop_reg(reg);
                            }
                            
                            let dest_loc = get_loc(&instr.dest);
                             store_op(&mut builder, dest_loc, 0);
                         }
                    }
                    Opcode::Ret => { 
                         if stack_size > 0 {
                             builder.add_rsp(stack_size);
                         }
                         builder.pop_reg(5); 
                         builder.pop_reg(10);
                         builder.pop_reg(9);
                         builder.pop_reg(8);
                         builder.pop_reg(7); 
                         builder.epilogue();
                    }
                    Opcode::Free => {
                         let free_addr = libc::free as usize as u64;
                         builder.mov_reg_imm64(0, free_addr);
                         if let Some(Operand::Reg(vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(vreg)).unwrap();
                             let s = load_op(&mut builder, src_loc, scratch1); 
                             builder.mov_rdi_reg(s); 
                         }
                         builder.push_reg(1); builder.push_reg(2); builder.push_reg(3); builder.push_reg(4);
                         builder.push_reg(6); builder.push_reg(11); builder.push_reg(12); builder.push_reg(13);
                         builder.call_reg(0);
                         builder.pop_reg(13); builder.pop_reg(12); builder.pop_reg(11); builder.pop_reg(6);
                         builder.pop_reg(4); builder.pop_reg(3); builder.pop_reg(2); builder.pop_reg(1);
                    }
                    Opcode::Alloc => {
                        let malloc_addr = libc::malloc as usize as u64;
                         builder.mov_reg_imm64(0, malloc_addr);
                         if let Some(Operand::Imm(val)) = instr.src1 {
                             builder.mov_rdi_imm(val);
                         } else if let Some(Operand::Reg(vreg)) = instr.src1 {
                             let src_loc = *gpr_map.get(&Operand::Reg(vreg)).unwrap();
                             let s = load_op(&mut builder, src_loc, scratch1);
                             builder.mov_rdi_reg(s);
                         }
                         builder.push_reg(1); builder.push_reg(2); builder.push_reg(3); builder.push_reg(4);
                         builder.push_reg(6); builder.push_reg(11); builder.push_reg(12); builder.push_reg(13);
                         builder.call_reg(0);
                         builder.pop_reg(13); builder.pop_reg(12); builder.pop_reg(11); builder.pop_reg(6);
                         builder.pop_reg(4); builder.pop_reg(3); builder.pop_reg(2); builder.pop_reg(1);
                         
                         let dest_loc = get_loc(&instr.dest);
                         store_op(&mut builder, dest_loc, 0);
                    }
                    Opcode::Load => {
                         let dest_loc = get_loc(&instr.dest);
                         let base_loc = get_loc(&instr.src1);
                         let base_reg = load_op(&mut builder, base_loc, scratch1);
                         
                         if let Some(Operand::Imm(idx)) = instr.src2 {
                             let d_reg = match dest_loc { Location::Register(r) => r, _ => scratch2 };
                             builder.mov_reg_imm(d_reg, idx);
                             builder.mov_reg_index(d_reg, base_reg, d_reg); 
                             if let Location::Spill(off) = dest_loc {
                                 builder.mov_stack_reg(off, d_reg);
                             }
                         } else if let Some(Operand::Reg(idx_vreg)) = instr.src2 {
                             let idx_loc = *gpr_map.get(&Operand::Reg(idx_vreg)).unwrap();
                             let idx_reg = load_op(&mut builder, idx_loc, scratch2); 
                             
                             let d_reg = match dest_loc { Location::Register(r) => r, _ => scratch1 }; 
                             builder.mov_reg_index(d_reg, base_reg, idx_reg);
                             if let Location::Spill(off) = dest_loc {
                                 builder.mov_stack_reg(off, d_reg);
                             }
                         }
                    }
                    Opcode::Store => {
                         let base_loc = get_loc(&instr.dest);
                         let base_reg = load_op(&mut builder, base_loc, scratch1);
                         let val_reg = if let Some(Operand::Imm(val)) = instr.src2 {
                             builder.mov_reg_imm(0, val); 
                             0
                         } else {
                             let v_loc = get_loc(&instr.src2);
                             load_op(&mut builder, v_loc, scratch2)
                         };
                         let idx_reg = if let Some(Operand::Imm(idx)) = instr.src1 {
                              builder.mov_reg_imm(6, idx);
                              6
                         } else {
                              let i_loc = get_loc(&instr.src1);
                              match i_loc {
                                  Location::Register(r) => r,
                                  Location::Spill(off) => { builder.mov_reg_stack(6, off); 6 }
                              }
                         };
                         builder.mov_index_reg(base_reg, idx_reg, val_reg);
                    }
                    _ => {} 
                }
            }

            builder.bind_label(&fail_label);
            builder.mov_reg_imm(0, -999);
            if stack_size > 0 { builder.add_rsp(stack_size); }
            builder.pop_reg(5);
            builder.pop_reg(10);
            builder.pop_reg(9);
            builder.pop_reg(8);
            builder.pop_reg(7);
            builder.epilogue();
        }

        let buf = builder.finalize();
        Ok((buf, main_offset))
    }
}

// Helper
fn is_caller_saved(r: u8) -> bool {
    matches!(r, 0 | 1 | 2 | 3 | 4 | 6 | 11 | 12 | 13)
}

fn liveness_analysis(func: &Function) -> Vec<Interval> {
    let mut starts = HashMap::new();
    let mut ends = HashMap::new();
    let mut ops = HashSet::new();
    let mut back_edges = Vec::new(); 
    let mut labels = HashMap::new();
    for (idx, instr) in func.instructions.iter().enumerate() {
        if instr.op == Opcode::Label {
            if let Some(Operand::Label(name)) = &instr.dest {
                labels.insert(name.clone(), idx);
            }
        }
    }
    for (idx, instr) in func.instructions.iter().enumerate() {
        if matches!(instr.op, Opcode::Jmp | Opcode::Jnz | Opcode::Je | Opcode::Jne | Opcode::Jl | Opcode::Jle | Opcode::Jg | Opcode::Jge) {
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
        for op in [&instr.dest, &instr.src1, &instr.src2].iter().filter_map(|x| x.as_ref()) {
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
    let mut intervals: Vec<Interval> = ops.into_iter().map(|op| {
        let start = *starts.get(&op).unwrap_or(&0);
        let mut end = *ends.get(&op).unwrap_or(&0);
        for &(loop_head, loop_tail) in &back_edges {
            if start <= loop_head && end >= loop_head {
                if end < loop_tail { end = loop_tail; }
            }
        }
        Interval { operand: op.clone(), start, end, assigned_loc: None }
    }).collect();
    intervals.sort_by_key(|i| i.start);
    intervals
}

fn allocate_registers(mut intervals: Vec<Interval>, pool: Vec<u8>, offset_start: i32) -> Result<(HashMap<Operand, Location>, i32), String> {
    let mut active: Vec<Interval> = Vec::new();
    let mut map = HashMap::new();
    let mut stack_slot_count = 0;

    for iv in &intervals {
         if let Operand::Reg(0) = iv.operand {
             map.insert(iv.operand.clone(), Location::Register(0));
         }
    }
    for r in 1..5 {
        let op = Operand::Reg(r);
        if intervals.iter().any(|i| i.operand == op) {
            map.insert(op, Location::Register(r));
        }
    }

    let mut pre_colored: HashMap<u8, Vec<Interval>> = HashMap::new();
    for iv in &intervals {
        if let Some(Location::Register(phys)) = map.get(&iv.operand) {
             pre_colored.entry(*phys).or_default().push(iv.clone());
        }
    }

    for i in 0..intervals.len() {
        let current_start = intervals[i].start;
        active.retain(|iv| iv.end > current_start);

        if map.contains_key(&intervals[i].operand) {
            intervals[i].assigned_loc = Some(map[&intervals[i].operand]);
            active.push(intervals[i].clone());
            continue;
        }

        let used_regs: HashSet<u8> = active.iter().filter_map(|iv| match iv.assigned_loc {
            Some(Location::Register(r)) => Some(r),
            _ => None
        }).collect();

        let mut free_regs: Vec<u8> = pool.iter().cloned()
            .filter(|r| !used_regs.contains(r))
            .filter(|r| {
                if let Some(fixed) = pre_colored.get(r) {
                     !fixed.iter().any(|f| intervals[i].start < f.end && f.start < intervals[i].end)
                } else { true }
            }).collect();
        free_regs.sort();

        if let Some(phys) = free_regs.first() {
            let loc = Location::Register(*phys);
            intervals[i].assigned_loc = Some(loc);
            map.insert(intervals[i].operand.clone(), loc);
            active.push(intervals[i].clone());
        } else {
            let spill_candidate_idx = active.iter()
                .enumerate()
                .max_by_key(|(_, iv)| iv.end)
                .map(|(idx, _)| idx);
            
            let must_spill_active = if let Some(idx) = spill_candidate_idx {
                active[idx].end > intervals[i].end
            } else { false };

            if must_spill_active {
                let idx = spill_candidate_idx.unwrap();
                let mut spilled_iv = active.remove(idx);
                let reg = match spilled_iv.assigned_loc {
                    Some(Location::Register(r)) => r,
                    _ => panic!("Active should be reg"),
                };
                
                stack_slot_count += 1;
                let offset = -(offset_start + stack_slot_count * 8); 
                let spill_loc = Location::Spill(offset);
                
                spilled_iv.assigned_loc = Some(spill_loc);
                map.insert(spilled_iv.operand.clone(), spill_loc);

                let loc = Location::Register(reg);
                intervals[i].assigned_loc = Some(loc);
                map.insert(intervals[i].operand.clone(), loc);
                active.push(intervals[i].clone());
            } else {
                 stack_slot_count += 1;
                let offset = -(offset_start + stack_slot_count * 8);
                let loc = Location::Spill(offset);
                intervals[i].assigned_loc = Some(loc);
                map.insert(intervals[i].operand.clone(), loc);
            }
        }
    }

    Ok((map, stack_slot_count))
}
