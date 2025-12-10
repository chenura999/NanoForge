use crate::jit_memory::DualMappedMemory;
use dynasmrt::{dynasm, x64::Assembler, DynamicLabel, DynasmApi, DynasmLabelApi};
use std::collections::HashMap;
use std::ptr;

pub struct CodeGenerator;

impl CodeGenerator {
    /// Generates a function that adds 'n' to its input argument.
    /// fn(x: i64) -> i64
    pub fn generate_add_n(n: i32) -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // System V AMD64 ABI:
        // arg1 (x) is in rdi
        // return value in rax
        dynasm!(ops
            ; .arch x64
            ; mov rax, rdi
            ; add rax, n
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates a function that sums numbers from 0 to n.
    /// fn(n: i64) -> i64
    pub fn generate_sum_loop() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // rdi = n
        // rax = sum (accumulator)
        // rcx = counter
        dynasm!(ops
            ; .arch x64
            ; mov rax, 0
            ; mov rcx, 0
            ; .align 16
            ; ->loop_start:
            ; cmp rcx, rdi
            ; jge ->loop_end
            ; add rax, rcx
            ; inc rcx
            ; jmp ->loop_start
            ; ->loop_end:
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates an unrolled version of the sum loop.
    /// This is a simple demonstration of structural optimization.
    pub fn generate_sum_loop_unrolled() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // rdi = n
        // rax = sum
        // rcx = counter
        dynasm!(ops
            ; .arch x64
            ; mov rax, 0
            ; mov rcx, 0
            ; .align 16
            ; ->loop_start:
            ; cmp rcx, rdi
            ; jge ->loop_end
            // Unroll 4 times
            ; add rax, rcx
            ; inc rcx
            ; cmp rcx, rdi
            ; jge ->loop_end
            ; add rax, rcx
            ; inc rcx
            ; cmp rcx, rdi
            ; jge ->loop_end
            ; add rax, rcx
            ; inc rcx
            ; cmp rcx, rdi
            ; jge ->loop_end
            ; add rax, rcx
            ; inc rcx

            ; jmp ->loop_start
            ; ->loop_end:
            ; ret
        );

        let buf = ops.finalize().unwrap();
        Ok(buf.to_vec())
    }

    /// Generates an AVX2 vectorized sum loop.
    /// Processes 8 integers per iteration.
    pub fn generate_sum_avx2() -> Result<Vec<u8>, String> {
        let mut ops = Assembler::new().unwrap();
        let _offset = ops.offset();

        // rdi = n
        // ymm0 = accumulator (zeros)
        // ymm1 = current vector [0, 1, 2, 3, 4, 5, 6, 7]
        // ymm2 = increment vector [8, 8, 8, 8, 8, 8, 8, 8]
        // rcx = counter (scalar)

        dynasm!(ops
            ; .arch x64
            ; vpxor ymm0, ymm0, ymm0        // ymm0 = 0

            // Create [0, 1, 2, 3, 4, 5, 6, 7] in ymm1
            // We push 64-bit values, each containing TWO 32-bit integers.
            // Stack grows down, so we push in reverse order.
            // We want memory: 0, 1, 2, 3, 4, 5, 6, 7
            // Push 4: (7, 6) -> 0x0000000700000006
            // Push 3: (5, 4) -> 0x0000000500000004
            // Push 2: (3, 2) -> 0x0000000300000002
            // Push 1: (1, 0) -> 0x0000000100000000

            ; mov rax, 0x0000000700000006; push rax
            ; mov rax, 0x0000000500000004; push rax
            ; mov rax, 0x0000000300000002; push rax
            ; mov rax, 0x0000000100000000; push rax
            ; vmovdqu ymm1, [rsp]
            ; add rsp, 32 // Clean up stack (4 * 8 bytes = 32)

            // Create [8, 8, ...] in ymm2
            // Each 64-bit push is (8, 8) -> 0x0000000800000008
            ; mov rax, 0x0000000800000008
            ; push rax; push rax; push rax; push rax
            ; vmovdqu ymm2, [rsp]
            ; add rsp, 32

            ; mov rcx, 0
            ; .align 16
            ; ->loop_start:
            ; cmp rcx, rdi
            ; jge ->loop_end

            ; vpaddd ymm0, ymm0, ymm1       // Accumulate
            ; vpaddd ymm1, ymm1, ymm2       // Increment indices

            ; add rcx, 8                    // Scalar increment by 8
            ; jmp ->loop_start
            ; ->loop_end:

            // Horizontal Sum ymm0 -> rax
            ; vextracti128 xmm1, ymm0, 1
            ; vpaddd xmm0, xmm0, xmm1
            ; vphaddd xmm0, xmm0, xmm0
            ; vphaddd xmm0, xmm0, xmm0
            ; vmovd eax, xmm0

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

// Helper to map NanoForge VReg to x64 HW Reg
fn get_hw_reg(r: u8) -> u8 {
    match r {
        0 => 0,   // RAX
        1 => 8,   // R8
        2 => 9,   // R9
        3 => 10,  // R10
        4 => 11,  // R11
        5 => 15,  // R15
        6 => 1,   // RCX
        7 => 3,   // RBX
        8 => 12,  // R12
        9 => 13,  // R13
        10 => 14, // R14
        11 => 7,  // RDI
        12 => 6,  // RSI
        13 => 2,  // RDX
        _ => panic!("Reg {} not mapped to HW", r),
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
        let ops = &mut self.ops;
        dynasm!(ops ; =>label);
    }

    pub fn current_offset(&self) -> usize {
        self.ops.offset().0
    }

    pub fn jmp(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jmp =>label);
    }

    pub fn jnz(&mut self, cond_reg: u8, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;

        match cond_reg {
            0 => dynasm!(ops ; .arch x64 ; test rax, rax),
            1 => dynasm!(ops ; .arch x64 ; test r8, r8),
            2 => dynasm!(ops ; .arch x64 ; test r9, r9),
            3 => dynasm!(ops ; .arch x64 ; test r10, r10),
            4 => dynasm!(ops ; .arch x64 ; test r11, r11),
            5 => dynasm!(ops ; .arch x64 ; test r15, r15),
            6 => dynasm!(ops ; .arch x64 ; test rcx, rcx),
            7 => dynasm!(ops ; .arch x64 ; test rbx, rbx),
            8 => dynasm!(ops ; .arch x64 ; test r12, r12),
            9 => dynasm!(ops ; .arch x64 ; test r13, r13),
            10 => dynasm!(ops ; .arch x64 ; test r14, r14),
            _ => panic!("Reg {} not supported for jnz", cond_reg),
        }
        dynasm!(ops ; .arch x64 ; jnz =>label);
    }

    pub fn cmp_reg_reg(&mut self, reg1: u8, reg2: u8) {
        let ops = &mut self.ops;
        let get_hw = |r: u8| -> u8 {
            match r {
                0 => 0,
                1 => 8,
                2 => 9,
                3 => 10,
                4 => 11,
                5 => 15,
                6 => 1,
                7 => 3,
                8 => 12,
                9 => 13,
                10 => 14,
                _ => panic!("Reg {}", r),
            }
        };
        let r1 = get_hw(reg1);
        let r2 = get_hw(reg2);
        dynasm!(ops ; .arch x64 ; cmp Rq(r1), Rq(r2));
    }

    pub fn cmp_reg_imm(&mut self, reg: u8, imm: i32) {
        let ops = &mut self.ops;
        match reg {
            0 => dynasm!(ops ; .arch x64 ; cmp rax, imm),
            1 => dynasm!(ops ; .arch x64 ; cmp r8, imm),
            2 => dynasm!(ops ; .arch x64 ; cmp r9, imm),
            3 => dynasm!(ops ; .arch x64 ; cmp r10, imm),
            4 => dynasm!(ops ; .arch x64 ; cmp r11, imm),
            5 => dynasm!(ops ; .arch x64 ; cmp r15, imm),
            6 => dynasm!(ops ; .arch x64 ; cmp rcx, imm),
            7 => dynasm!(ops ; .arch x64 ; cmp rbx, imm),
            8 => dynasm!(ops ; .arch x64 ; cmp r12, imm),
            9 => dynasm!(ops ; .arch x64 ; cmp r13, imm),
            10 => dynasm!(ops ; .arch x64 ; cmp r14, imm),
            _ => panic!("Cmp {}, imm not supported", reg),
        }
    }

    pub fn je(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; je =>label);
    }

    pub fn jne(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jne =>label);
    }

    pub fn jl(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jl =>label);
    }

    pub fn jle(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jle =>label);
    }

    pub fn jg(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jg =>label);
    }

    pub fn jge(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; jge =>label);
    }

    pub fn call(&mut self, name: &str) {
        let label = self.get_label(name);
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; call =>label);
    }

    // ... existing math ops ...
    pub fn add_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let ops = &mut self.ops;
        match dest_reg {
            0 => dynasm!(ops ; .arch x64 ; add rax, imm),
            1 => dynasm!(ops ; .arch x64 ; add r8, imm),
            2 => dynasm!(ops ; .arch x64 ; add r9, imm),
            3 => dynasm!(ops ; .arch x64 ; add r10, imm),
            4 => dynasm!(ops ; .arch x64 ; add r11, imm),
            5 => dynasm!(ops ; .arch x64 ; add r15, imm),
            6 => dynasm!(ops ; .arch x64 ; add rcx, imm),
            7 => dynasm!(ops ; .arch x64 ; add rbx, imm),
            8 => dynasm!(ops ; .arch x64 ; add r12, imm),
            9 => dynasm!(ops ; .arch x64 ; add r13, imm),
            10 => dynasm!(ops ; .arch x64 ; add r14, imm),
            _ => panic!("Add Reg {} not supported", dest_reg),
        }
    }
    pub fn sub_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        dynasm!(ops ; .arch x64 ; sub Rq(d), imm);
    }

    pub fn mov_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        dynasm!(ops ; .arch x64 ; mov Rd(d), imm);
    }

    pub fn mov_reg_imm64(&mut self, dest_reg: u8, imm: u64) {
        let ops = &mut self.ops;
        let imm_val = imm as i64;
        let d = get_hw_reg(dest_reg);
        dynasm!(ops ; .arch x64 ; mov Rq(d), QWORD imm_val);
    }

    pub fn mov_reg_stack(&mut self, dest_reg: u8, offset: i32) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        dynasm!(ops ; .arch x64 ; mov Rq(d), [rbp + offset]);
    }

    pub fn mov_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        let s = get_hw_reg(src_reg);
        dynasm!(ops ; .arch x64 ; mov Rq(d), Rq(s));
    }

    pub fn add_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        let s = get_hw_reg(src_reg);
        dynasm!(ops ; .arch x64 ; add Rq(d), Rq(s));
    }

    pub fn sub_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        let s = get_hw_reg(src_reg);
        dynasm!(ops ; .arch x64 ; sub Rq(d), Rq(s));
    }

    // AVX2 Instructions
    // VLoad: vmovdqu ymm, [base + index*8] (Wait, index*8 is for 64-bit pointers)
    // Here we load 32 bytes (256 bits).
    // The address calculation is standard SIB: [base + index * 1 (or 4?)]
    // In NanoForge, all pointers are 64-bit aligned. `Alloc` returns `u64` (ptr).
    // `A[i]` means `*(A + i*4)`? No, NanoForge likely treats `i` as index into what?
    // In `parser.rs`: `Load` -> `dest = [base + index * 8]`.
    // So NanoForge arrays are 64-bit integers (stride 8).
    // AVX2 can load 4 x 64-bit integers (256 bits = 4 qwords).
    // So stride for vector load should be 4 elements = 32 bytes.
    // Address is strictly `base + index*8`.

    // `vmovdqu destination, source`

    pub fn vmovdqu_load(
        &mut self,
        dest_ymm: u8,
        base_reg: u8,
        index_reg: u8,
        offset_elements: i32,
    ) {
        let ops = &mut self.ops;
        let b = get_hw_reg(base_reg);
        let i = get_hw_reg(index_reg);
        let y = dest_ymm;
        let disp = offset_elements * 8;
        dynasm!(ops ; .arch x64 ; vmovdqu Ry(y), [Rq(b) + Rq(i) * 8 + disp]);
    }

    pub fn vmovdqu_store(
        &mut self,
        base_reg: u8,
        index_reg: u8,
        src_ymm: u8,
        offset_elements: i32,
    ) {
        let ops = &mut self.ops;
        let b = get_hw_reg(base_reg);
        let i = get_hw_reg(index_reg);
        let y = src_ymm;
        let disp = offset_elements * 8;
        dynasm!(ops ; .arch x64 ; vmovdqu [Rq(b) + Rq(i) * 8 + disp], Ry(y));
    }

    pub fn vpaddq(&mut self, dest_ymm: u8, src1_ymm: u8, src2_ymm: u8) {
        let ops = &mut self.ops;
        let d = dest_ymm;
        let s1 = src1_ymm;
        let s2 = src2_ymm;
        dynasm!(ops ; .arch x64 ; vpaddq Ry(d), Ry(s1), Ry(s2));
    }

    pub fn mov_reg_index(&mut self, dest_reg: u8, base_reg: u8, index_reg: u8) {
        let ops = &mut self.ops;
        let d = get_hw_reg(dest_reg);
        let b = get_hw_reg(base_reg);
        let i = get_hw_reg(index_reg);
        dynasm!(ops ; .arch x64 ; mov Rq(d), [Rq(b) + Rq(i) * 8]);
    }

    pub fn mov_index_reg(&mut self, base_reg: u8, index_reg: u8, src_reg: u8) {
        let ops = &mut self.ops;
        let b = get_hw_reg(base_reg);
        let i = get_hw_reg(index_reg);
        let s = get_hw_reg(src_reg);
        dynasm!(ops ; .arch x64 ; mov [Rq(b) + Rq(i) * 8], Rq(s));
    }

    pub fn call_reg(&mut self, reg: u8) {
        let ops = &mut self.ops;
        let r = get_hw_reg(reg);
        dynasm!(ops ; .arch x64 ; call Rq(r));
    }

    pub fn push_reg(&mut self, reg: u8) {
        let ops = &mut self.ops;
        let r = get_hw_reg(reg);
        dynasm!(ops ; .arch x64 ; push Rq(r));
    }

    pub fn pop_reg(&mut self, reg: u8) {
        let ops = &mut self.ops;
        let r = get_hw_reg(reg);
        dynasm!(ops ; .arch x64 ; pop Rq(r));
    }

    pub fn prologue(&mut self, stack_size: i32) {
        let ops = &mut self.ops;
        let aligned_size = (stack_size + 15) & !15;

        dynasm!(ops
            ; .arch x64
            ; push rbp
            ; mov rbp, rsp
            // Save Callee-Saved Registers:
            // R15 (Reg 5), RBX (Reg 7), R12 (8), R13 (9), R14 (10).
            // 5 registers * 8 = 40 bytes.
            // + RBP (8) + RetAddr (8) = 56 bytes.
            // 56 % 16 = 8. Misaligned.
            // We need to push 1 more or sub rsp, 8.
            // Let's push one more? No, let's use sub.

            ; push r15
            ; push rbx
            ; push r12
            ; push r13
            ; push r14

            ; sub rsp, 8 // Align to 16
        );

        if aligned_size > 0 {
            dynasm!(ops ; .arch x64 ; sub rsp, aligned_size);
        }
    }

    pub fn add_rsp(&mut self, offset: i32) {
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; add rsp, offset);
    }

    pub fn epilogue(&mut self) {
        let ops = &mut self.ops;
        dynasm!(ops
            ; .arch x64
            ; lea rsp, [rbp - 40] // Point to R14 (Bottom of saved regs)
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; pop r15
            ; pop rbp
            ; ret
        );
    }

    pub fn mov_rdi_imm(&mut self, imm: i32) {
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; mov rdi, imm);
    }

    pub fn mov_rdi_reg(&mut self, src_reg: u8) {
        let ops = &mut self.ops;
        let s = get_hw_reg(src_reg);
        dynasm!(ops ; .arch x64 ; mov rdi, Rq(s));
    }

    pub fn rdtsc(&mut self) {
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; rdtsc);
    }

    pub fn ret(&mut self) {
        let ops = &mut self.ops;
        dynasm!(ops ; ret);
    }

    pub fn finalize(self) -> Vec<u8> {
        self.ops.finalize().unwrap().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_memory::DualMappedMemory;

    #[test]
    fn test_avx2_sum_loop() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test: AVX2 not supported on this host.");
            return;
        }

        let code = CodeGenerator::generate_sum_avx2().expect("Failed to generate AVX2 code");
        let memory = DualMappedMemory::new(4096).expect("Failed to allocate memory");

        CodeGenerator::emit_to_memory(&memory, &code, 0);

        let func: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };

        let n = 1000;
        let result = func(n);
        let expected: i64 = (0..n).sum();

        assert_eq!(result, expected, "AVX2 sum loop failed");
    }
}
