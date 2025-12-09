use std::arch::x86_64::_rdtsc;

pub struct Benchmarker;

impl Benchmarker {
    /// Measures the average CPU cycles taken by a function over `iterations`.
    ///
    /// # Safety
    /// This function executes arbitrary code generated at runtime.
    /// The caller must ensure the function pointer is valid.
    pub unsafe fn measure(func: extern "C" fn(u64) -> u64, input: u64, iterations: u64) -> u64 {
        // Warmup
        for _ in 0..100 {
            func(input);
        }

        let start = _rdtsc();
        for _ in 0..iterations {
            func(input);
        }
        let end = _rdtsc();

        (end - start) / iterations
    }
}
