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
            ; add rax, n as i32
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

    /// Writes the generated code into the DualMappedMemory at the specified offset.
    pub fn emit_to_memory(memory: &DualMappedMemory, code: &[u8], offset: usize) {
        unsafe {
            let dest = memory.rw_ptr.add(offset);
            ptr::copy_nonoverlapping(code.as_ptr(), dest, code.len());
        }
        memory.flush_icache();
    }
}
