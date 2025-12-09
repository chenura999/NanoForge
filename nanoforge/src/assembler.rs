use crate::jit_memory::DualMappedMemory;
use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
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
