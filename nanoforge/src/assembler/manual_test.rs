#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_memory::DualMappedMemory;
    use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};
    use std::mem;

    #[test]
    fn test_manual_register_integrity() {
        let mut ops = dynasmrt::x64::Assembler::new().unwrap();
        let offset = ops.offset();

        // Test R13 (Current Reg 1)
        // mov r13d, 42
        // mov eax, r13d
        // ret
        dynasm!(ops
            ; .arch x64
            ; mov r13d, 42
            ; mov eax, r13d
            ; ret
        );

        let buf = ops.finalize().unwrap();
        let memory = DualMappedMemory::new(4096).unwrap();

        // Copy to executable memory
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), memory.rw_ptr, buf.len());
        }

        let func: extern "C" fn() -> i32 = unsafe { mem::transmute(memory.rx_ptr) };
        let result = func();

        println!("Manual ASM R13 Result: {}", result);
        assert_eq!(result, 42, "R13 failed to hold value in manual assembly");
    }
}
