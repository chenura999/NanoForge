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

        // test reg, reg to check for zero
        match cond_reg {
            0 => dynasm!(ops ; .arch x64 ; test rax, rax),
            1 => dynasm!(ops ; .arch x64 ; test r8, r8),
            2 => dynasm!(ops ; .arch x64 ; test r9, r9),
            3 => dynasm!(ops ; .arch x64 ; test r10, r10),
            4 => dynasm!(ops ; .arch x64 ; test r11, r11),
            5 => dynasm!(ops ; .arch x64 ; test r15, r15),
            _ => panic!("Reg {} not supported for jnz", cond_reg),
        }
        dynasm!(ops ; .arch x64 ; jnz =>label);
    }

    pub fn cmp_reg_reg(&mut self, reg1: u8, reg2: u8) {
        let ops = &mut self.ops;
        let get_hw = |r: u8| -> u8 {
            match r {
                0 => 0, // RAX
                1 => 8, // R8
                2 => 9, // R9
                3 => 10,
                4 => 11,
                5 => 15,
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
            _ => panic!("Add Reg {} not supported", dest_reg),
        }
    }
    pub fn sub_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        println!("ASM: sub reg{}, {}", dest_reg, imm);
        let ops = &mut self.ops;
        match dest_reg {
            0 => dynasm!(ops ; .arch x64 ; sub rax, imm),
            1 => dynasm!(ops ; .arch x64 ; sub r8, imm),
            2 => dynasm!(ops ; .arch x64 ; sub r9, imm),
            3 => dynasm!(ops ; .arch x64 ; sub r10, imm),
            4 => dynasm!(ops ; .arch x64 ; sub r11, imm),
            5 => dynasm!(ops ; .arch x64 ; sub r15, imm),
            _ => panic!("Reg {} not supported", dest_reg),
        }
    }

    pub fn mov_reg_imm(&mut self, dest_reg: u8, imm: i32) {
        let ops = &mut self.ops;
        println!("ASM: mov reg{}, {}", dest_reg, imm);
        match dest_reg {
            0 => dynasm!(ops ; .arch x64 ; mov eax, imm),
            1 => dynasm!(ops ; .arch x64 ; mov r8d, imm),
            2 => dynasm!(ops ; .arch x64 ; mov r9d, imm),
            3 => dynasm!(ops ; .arch x64 ; mov r10d, imm),
            4 => dynasm!(ops ; .arch x64 ; mov r11d, imm),
            5 => dynasm!(ops ; .arch x64 ; mov r15d, imm),
            _ => panic!("Register {} not supported", dest_reg),
        }
    }

    pub fn mov_reg_stack(&mut self, dest_reg: u8, offset: i32) {
        let ops = &mut self.ops;
        let get_hw = |r: u8| -> u8 {
            match r {
                0 => 0,  // RAX
                1 => 8,  // R8
                2 => 9,  // R9
                3 => 10, // R10
                4 => 11, // R11
                5 => 15, // R15
                _ => panic!("Register {} not supported", r),
            }
        };
        let d = get_hw(dest_reg);
        dynasm!(ops ; .arch x64 ; mov Rd(d), [rbp + offset]);
    }

    pub fn mov_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        let ops = &mut self.ops;
        let get_hw = |r: u8| -> u8 {
            match r {
                0 => 0,  // RAX
                1 => 8,  // R8
                2 => 9,  // R9
                3 => 10, // R10
                4 => 11, // R11
                5 => 15, // R15
                _ => panic!("Register {} not supported", r),
            }
        };
        let d = get_hw(dest_reg);
        let s = get_hw(src_reg);
        dynasm!(ops ; .arch x64 ; mov Rq(d), Rq(s));
    }

    pub fn add_reg_reg(&mut self, dest_reg: u8, src_reg: u8) {
        println!("ASM: add reg{}, reg{}", dest_reg, src_reg);
        let ops = &mut self.ops;
        let get_hw = |r: u8| -> u8 {
            match r {
                0 => 0,  // RAX
                1 => 8,  // R8
                2 => 9,  // R9
                3 => 10, // R10
                4 => 11, // R11
                5 => 15, // R15
                _ => panic!("Reg {}", r),
            }
        };
        let d = get_hw(dest_reg);
        let s = get_hw(src_reg);
        dynasm!(ops ; .arch x64 ; add Rq(d), Rq(s));
    }

    pub fn push_reg(&mut self, reg: u8) {
        let ops = &mut self.ops;
        match reg {
            0 => dynasm!(ops ; .arch x64 ; push rax),
            1 => dynasm!(ops ; .arch x64 ; push r8),
            2 => dynasm!(ops ; .arch x64 ; push r9),
            3 => dynasm!(ops ; .arch x64 ; push r10),
            4 => dynasm!(ops ; .arch x64 ; push r11),
            5 => dynasm!(ops ; .arch x64 ; push r15),
            _ => panic!("Register {} not supported", reg),
        }
    }

    pub fn pop_reg(&mut self, reg: u8) {
        let ops = &mut self.ops;
        match reg {
            0 => dynasm!(ops ; .arch x64 ; pop rax),
            1 => dynasm!(ops ; .arch x64 ; pop r8),
            2 => dynasm!(ops ; .arch x64 ; pop r9),
            3 => dynasm!(ops ; .arch x64 ; pop r10),
            4 => dynasm!(ops ; .arch x64 ; pop r11),
            5 => dynasm!(ops ; .arch x64 ; pop r15),
            _ => panic!("Register {} not supported", reg),
        }
    }

    pub fn prologue(&mut self, stack_size: i32) {
        println!("Emitting Prologue. Size: {}", stack_size);
        let ops = &mut self.ops;
        dynasm!(ops
            ; push rbp
            ; mov rbp, rsp
            ; sub rsp, 16 // Hardcoded for test
            // Save Registers we use (R8..R11, R15)
            // Even though R8..R11 are Caller-Saved in SysV, our internal ABI
            // Use Rq() explicit codes to ensure REX prefixes
            ; push Rq(8)  // R8
            ; push Rq(9)  // R9
            ; push Rq(10) // R10
            ; push Rq(11) // R11
            ; push Rq(15) // R15
        );
    }

    pub fn add_rsp(&mut self, offset: i32) {
        let ops = &mut self.ops;
        dynasm!(ops ; .arch x64 ; add rsp, offset);
    }

    pub fn epilogue(&mut self) {
        let ops = &mut self.ops;
        dynasm!(ops
            // Restore Registers (Reverse)
            ; pop Rq(15)
            ; pop Rq(11)
            ; pop Rq(10)
            ; pop Rq(9)
            ; pop Rq(8)
            ; mov rsp, rbp
            ; pop rbp
            ; ret
        );
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
