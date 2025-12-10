use crate::jit_memory::DualMappedMemory;
use dynasmrt::{aarch64::Assembler, dynasm, DynamicLabel, DynasmApi, DynasmLabelApi};
use std::collections::HashMap;
use std::ptr;

pub struct CodeGenerator;

impl CodeGenerator {
    /// Generates a function that adds 'n' to its input argument.
    /// fn(x: i64) -> i64
    /// ARM64: arg1 in x0, return in x0
    pub fn generate_add_n(n: i32) -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // Check if n fits in immediate encoding, otherwise load it.
        // For simplicity in this demo, we'll assume it handles standard immediates or we'd move to reg.
        // dynasm-rs aarch64 backend handles some immediates, but large ones need explicit loading.
        // For 'add' immediate, it's 12-bit possibly shifted.

        dynasm!(ops
            ; .arch aarch64
            // x0 is argument and return register.
            // We need to add n to x0.
            ; mov x1, n as u64
            ; add x0, x0, x1
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates a function that sums numbers from 0 to n.
    /// fn(n: i64) -> i64
    /// Loop 0..n
    pub fn generate_sum_loop() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // x0 = n (limit)
        // x1 = accumulator (sum)
        // x2 = counter

        dynasm!(ops
            ; .arch aarch64
            ; mov x1, 0          // sum = 0
            ; mov x2, 0          // counter = 0

            ; ->loop_start:
            ; cmp x2, x0
            ; b.ge ->loop_end

            ; add x1, x1, x2     // sum += counter
            ; add x2, x2, 1      // counter++
            ; b ->loop_start

            ; ->loop_end:
            ; mov x0, x1         // return sum (in x0)
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates an unrolled version of the sum loop.
    pub fn generate_sum_loop_unrolled() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        dynasm!(ops
            ; .arch aarch64
            ; mov x1, 0
            ; mov x2, 0

            ; ->loop_start:
            ; cmp x2, x0
            ; b.ge ->loop_end

            // Unroll 4 times
            ; add x1, x1, x2; add x2, x2, 1
            ; cmp x2, x0; b.ge ->loop_end

            ; add x1, x1, x2; add x2, x2, 1
            ; cmp x2, x0; b.ge ->loop_end

            ; add x1, x1, x2; add x2, x2, 1
            ; cmp x2, x0; b.ge ->loop_end

            ; add x1, x1, x2; add x2, x2, 1

            ; b ->loop_start
            ; ->loop_end:
            ; mov x0, x1
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates a NEON vectorized sum loop.
    /// Comparable to AVX2, processes 4 x 32-bit (or 2x64) per vector.
    /// NEON registers are q0-q31 (128-bit).
    pub fn generate_sum_neon() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // x0 = n
        // v0 = accumulator (zeros)
        // v1 = current vector [0, 1, 2, 3] (32-bit lanes) -> 128 bit = 4 integers
        // v2 = increment vector [4, 4, 4, 4]
        // x3 = counter scalar

        // Setup constants:
        // We'll use 32-bit ints for simplicity to pack 4.

        dynasm!(ops
            ; .arch aarch64

            // Init accumulator v0 = 0
            ; movi v0.4s, 0

            // Init current vector v1 = {0, 1, 2, 3}
            // We have to load this.
            // Aarch64 literal pools are tricky in raw assembler without helpers.
            // We carefully construct it in GPRs and move to V regs.
            // 0, 1, 2, 3
            // 0x00000001_00000000 -> x4
            // 0x00000003_00000002 -> x5

            ; mov x4, 0x00000000
            ; movk x4, 0x0001, lsl 32

            ; mov x5, 0x0002
            ; movk x5, 0x0003, lsl 32

            // Move x4, x5 to v1 (q1)
            // vmov can move general purpose to vector element.
            // ins v1.d[0], x4
            // ins v1.d[1], x5
            ; ins v1.d[0], x4
            ; ins v1.d[1], x5

            // Init increment vector v2 = {4, 4, 4, 4}
            ; movi v2.4s, 4

            ; mov x3, 0 // Scalar counter

            ; ->loop_start:
            ; cmp x3, x0
            ; b.ge ->loop_end

            ; add v0.4s, v0.4s, v1.4s  // Accumulate
            ; add v1.4s, v1.4s, v2.4s  // Increment indices

            ; add x3, x3, 4            // Scalar increment
            ; b ->loop_start

            ; ->loop_end:
            // Horizontal sum v0 -> x0 (return)
            // addv s0, v0.4s (Add across vector into scalar register s0)
            ; addv s0, v0.4s
            ; fmov w0, s0  // Move float scalar to int w0
            // Implicitly x0 has the value zero-extended (or just w0 is fine for 32 bit sum)
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Writes the generated code into the DualMappedMemory at the specified offset.
    pub fn emit_to_memory(memory: &DualMappedMemory, code: &[u8], offset: usize) {
        unsafe {
            let dest = memory.rw_ptr.add(offset);
            ptr::copy_nonoverlapping(code.as_ptr(), dest, code.len());
        }
        memory.flush_icache();
    }
}

pub struct JitBuilder {
    ops: Assembler,
    labels: HashMap<String, DynamicLabel>,
}

impl JitBuilder {
    pub fn new() -> Self {
        Self {
            ops: Assembler::new().unwrap(),
            labels: HashMap::new(),
        }
    }

    fn get_label(&mut self, name: &str) -> DynamicLabel {
        if let Some(&label) = self.labels.get(name) {
            label
        } else {
            let label = self.ops.new_dynamic_label();
            self.labels.insert(name.to_string(), label);
            label
        }
    }

    pub fn bind_label(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; =>label);
    }

    pub fn jmp(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b =>label);
    }

    pub fn jnz(&mut self, cond_reg: u8, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        match cond_reg {
            0 => dynasm!(ops ; .arch aarch64 ; cbnz x0, =>label),
            1 => dynasm!(ops ; .arch aarch64 ; cbnz x1, =>label),
            2 => dynasm!(ops ; .arch aarch64 ; cbnz x2, =>label),
            _ => panic!("Reg {} not supported for jnz", cond_reg),
        }
    }

    pub fn cmp_reg_reg(&mut self, reg1: u8, reg2: u8) {
        let mut ops = &mut self.ops;
        match (reg1, reg2) {
            (0, 1) => dynasm!(ops ; .arch aarch64 ; cmp x0, x1),
            (0, 2) => dynasm!(ops ; .arch aarch64 ; cmp x0, x2),
            (1, 0) => dynasm!(ops ; .arch aarch64 ; cmp x1, x0),
            (1, 2) => dynasm!(ops ; .arch aarch64 ; cmp x1, x2),
            (2, 0) => dynasm!(ops ; .arch aarch64 ; cmp x2, x0),
            (2, 1) => dynasm!(ops ; .arch aarch64 ; cmp x2, x1),
            _ => panic!("Cmp {}, {} not supported", reg1, reg2),
        }
    }

    pub fn cmp_reg_imm(&mut self, reg: u8, imm: i32) {
        let mut ops = &mut self.ops;
        match reg {
            0 => dynasm!(ops ; .arch aarch64 ; cmp x0, imm as u64),
            1 => dynasm!(ops ; .arch aarch64 ; cmp x1, imm as u64),
            2 => dynasm!(ops ; .arch aarch64 ; cmp x2, imm as u64),
            _ => panic!("Cmp {}, imm not supported", reg),
        }
    }

    pub fn je(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.eq =>label);
    }
    pub fn jne(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.ne =>label);
    }
    pub fn jl(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.lt =>label);
    }
    pub fn jle(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.le =>label);
    }
    pub fn jg(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.gt =>label);
    }
    pub fn jge(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; b.ge =>label);
    }

    pub fn call(&mut self, name: &str) {
        let label = self.get_label(name);
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; bl =>label);
    }

    pub fn sub_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let mut ops = &mut self.ops;
        match dest_reg {
            0 => dynasm!(ops ; .arch aarch64 ; sub x0, x0, imm as u64),
            1 => dynasm!(ops ; .arch aarch64 ; sub x1, x1, imm as u64),
            2 => dynasm!(ops ; .arch aarch64 ; sub x2, x2, imm as u64),
            _ => panic!("Reg {} not supported", dest_reg),
        }
    }

    pub fn mov_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let mut ops = &mut self.ops;
        // x0, x1, x2 ...
        match dest_reg {
            0 => dynasm!(ops ; .arch aarch64 ; mov x0, imm as u64),
            1 => dynasm!(ops ; .arch aarch64 ; mov x1, imm as u64),
            2 => dynasm!(ops ; .arch aarch64 ; mov x2, imm as u64),
            _ => panic!("Reg {} not supported", dest_reg),
        }
    }

    pub fn mov_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let mut ops = &mut self.ops;
        match (dest_reg, src_reg) {
            (0, 1) => dynasm!(ops ; .arch aarch64 ; mov x0, x1),
            (0, 2) => dynasm!(ops ; .arch aarch64 ; mov x0, x2),
            (1, 0) => dynasm!(ops ; .arch aarch64 ; mov x1, x0),
            (1, 2) => dynasm!(ops ; .arch aarch64 ; mov x1, x2),
            (2, 0) => dynasm!(ops ; .arch aarch64 ; mov x2, x0),
            (2, 1) => dynasm!(ops ; .arch aarch64 ; mov x2, x1),
            _ => panic!("Mov {}, {} not supported", dest_reg, src_reg),
        }
    }

    pub fn add_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let mut ops = &mut self.ops;
        match (dest_reg, src_reg) {
            (0, 1) => dynasm!(ops ; .arch aarch64 ; add x0, x0, x1),
            (0, 2) => dynasm!(ops ; .arch aarch64 ; add x0, x0, x2),
            (1, 2) => dynasm!(ops ; .arch aarch64 ; add x1, x1, x2),
            (2, 1) => dynasm!(ops ; .arch aarch64 ; add x2, x2, x1),
            _ => panic!("Add {}, {} not supported", dest_reg, src_reg),
        }
    }

    pub fn push_reg(&mut self, reg: u8) {
        let mut ops = &mut self.ops;
        // Stack must be 16-byte aligned.
        // str xR, [sp, -16]!
        match reg {
            0 => dynasm!(ops ; .arch aarch64 ; str x0, [sp, -16]!),
            1 => dynasm!(ops ; .arch aarch64 ; str x1, [sp, -16]!),
            2 => dynasm!(ops ; .arch aarch64 ; str x2, [sp, -16]!),
            _ => panic!("Push reg {} not impl", reg),
        }
    }

    pub fn pop_reg(&mut self, reg: u8) {
        let mut ops = &mut self.ops;
        match reg {
            0 => dynasm!(ops ; .arch aarch64 ; ldr x0, [sp], 16),
            1 => dynasm!(ops ; .arch aarch64 ; ldr x1, [sp], 16),
            2 => dynasm!(ops ; .arch aarch64 ; ldr x2, [sp], 16),
            _ => panic!("Pop reg {} not impl", reg),
        }
    }

    pub fn prologue(&mut self, stack_size: i32) {
        let mut ops = &mut self.ops;
        // Save FP and LR
        dynasm!(ops
            ; .arch aarch64
            ; stp x29, x30, [sp, -16]!
            ; mov x29, sp
        );
        if stack_size > 0 {
            // align to 16
            let aligned = (stack_size + 15) & !15;
            dynasm!(ops ; .arch aarch64 ; sub sp, sp, aligned);
        }
    }

    pub fn epilogue(&mut self) {
        let mut ops = &mut self.ops;
        dynasm!(ops
            ; .arch aarch64
            ; mov sp, x29
            ; ldp x29, x30, [sp], 16
            ; ret
        );
    }

    pub fn ret(&mut self) {
        let mut ops = &mut self.ops;
        dynasm!(ops ; .arch aarch64 ; ret);
    }

    pub fn finalize(self) -> Vec<u8> {
        self.ops.finalize().unwrap().to_vec()
    }
}
