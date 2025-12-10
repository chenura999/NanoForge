#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::x64::JitBuilder;
    use crate::jit_memory::DualMappedMemory;
    use std::mem;

    #[test]
    fn test_rdtsc() {
        let mut assembler = JitBuilder::new();

        // rdtsc returns result in EDX:EAX (High:Low)
        // We want to return RAX loop count.
        assembler.rdtsc();
        // Note: rdtsc clobbers RDX and RAX.
        // Our function returns u32 via RAX (EAX).
        // Since we return u32, we just read EAX.

        assembler.ret();

        let code = assembler.finalize();
        let memory = DualMappedMemory::new(4096).unwrap();
        crate::assembler::CodeGenerator::emit_to_memory(&memory, &code, 0);

        let func: extern "C" fn() -> u32 = unsafe { mem::transmute(memory.rx_ptr) };
        let t1 = func();
        // Busy wait a bit? No, just call again.
        for _ in 0..1000 {
            std::hint::spin_loop();
        }
        let t2 = func();
        println!("T1: {}, T2: {}", t1, t2);
        assert!(t2 > t1, "TSC should increase");
        assert!(t1 > 0, "TSC should be non-zero");
    }
}
