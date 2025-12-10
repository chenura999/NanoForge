use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::ptr;

pub struct DualMappedMemory {
    pub rw_ptr: *mut u8,
    pub rx_ptr: *const u8,
    pub size: usize,
    fd: RawFd,
}

impl DualMappedMemory {
    pub fn new(size: usize) -> Result<Self, String> {
        unsafe {
            // 1. Create an anonymous file in memory
            let name = CString::new("nanoforge_jit").unwrap();
            let fd = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                return Err("memfd_create failed".to_string());
            }

            // 2. Set the size
            if libc::ftruncate(fd, size as i64) < 0 {
                libc::close(fd);
                return Err("ftruncate failed".to_string());
            }

            // 3. Map as Read-Write (The "Writer" View)
            let rw_ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if rw_ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err("mmap RW failed".to_string());
            }

            // 4. Map as Read-Execute (The "Executor" View)
            let rx_ptr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if rx_ptr == libc::MAP_FAILED {
                libc::munmap(rw_ptr, size);
                libc::close(fd);
                return Err("mmap RX failed".to_string());
            }

            Ok(DualMappedMemory {
                rw_ptr: rw_ptr as *mut u8,
                rx_ptr: rx_ptr as *const u8,
                size,
                fd,
            })
        }
    }

    /// Flushes the Instruction Cache for the allocated memory.
    /// This ensures that the CPU sees the new instructions we just wrote.
    pub fn flush_icache(&self) {
        unsafe {
            // __builtin___clear_cache is a GCC/Clang intrinsic.
            // In Rust, we can use the unstable std::intrinsics or just call a C function.
            // However, libc doesn't always expose it.
            // For x86_64, strictly speaking, hardware handles coherency, but 'clflush' is good practice.
            // A portable way in Rust is hard without nightly.
            // We will use a simple assembly block for x86_64 to serialize.

            #[cfg(target_arch = "x86_64")]
            {
                // mfence is sufficient to drain store buffers.
                // For full serialization, cpuid is needed, but rbx is reserved by LLVM.
                // We'll just use mfence for this PoC to avoid complexity.
                std::arch::asm!("mfence", options(nostack));
            }

            #[cfg(target_arch = "aarch64")]
            {
                // Aarch64 Cache Coherency:
                // 1. Clean data cache by VA to PoU (Point of Unification)
                // 2. Invalidate instruction cache by VA to PoU
                // 3. ISB (Instruction Synchronization Barrier) to ensure fetch pipeline sees it.

                let start = self.rx_ptr as usize;
                let end = start + self.size;
                // Cache line size is usually 64 bytes (CTR_EL0), but we'll iterate.
                // Ideally reading lookup size is better, but step of 64 is safe on modern ARM64.
                // Or we can rely on system primitives.
                // For this PoC, we do a loop.

                let stride = 64;
                let mut addr = start;
                while addr < end {
                    // DC CVAU: Data Cache Clean by VA to Point of Unification
                    std::arch::asm!("dc cvau, {0}", in(reg) addr);
                    addr += stride;
                }

                std::arch::asm!("dsb ish"); // Data Synchronization Barrier (Inner Shareable)

                addr = start;
                while addr < end {
                    // IC IVAU: Instruction Cache Invalidate by VA to Point of Unification
                    std::arch::asm!("ic ivau, {0}", in(reg) addr);
                    addr += stride;
                }

                std::arch::asm!("dsb ish"); // Ensure IC invalidation completes
                std::arch::asm!("isb"); // Instruction Synchronization Barrier (Flush pipeline)
            }

            // Ideally we would use:
            // extern "C" { fn __clear_cache(start: *mut c_void, end: *mut c_void); }
            // __clear_cache(self.rx_ptr as *mut _, self.rx_ptr.add(self.size) as *mut _);
        }
    }
}

// SAFETY: We are responsible for ensuring no data races occur.
// The RW view is only used during initialization (before publishing).
// The RX view is read-only after publishing.
unsafe impl Send for DualMappedMemory {}
unsafe impl Sync for DualMappedMemory {}

impl Drop for DualMappedMemory {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.rw_ptr as *mut _, self.size);
            libc::munmap(self.rx_ptr as *mut _, self.size);
            libc::close(self.fd);
        }
    }
}
