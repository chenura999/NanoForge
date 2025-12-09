use std::mem;
use std::time::Instant;

pub fn benchmark_function(func: extern "C" fn(i64) -> i64, input: i64, iterations: u64) -> u128 {
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(func(input));
    }
    start.elapsed().as_nanos()
}

pub fn pin_thread_to_core(core_id: usize) -> Result<(), String> {
    unsafe {
        let mut cpuset: libc::cpu_set_t = mem::zeroed();
        libc::CPU_SET(core_id, &mut cpuset);

        let pid = 0; // 0 means current thread
        let ret = libc::sched_setaffinity(pid, mem::size_of::<libc::cpu_set_t>(), &cpuset);

        if ret != 0 {
            return Err(format!("Failed to pin thread to core {}", core_id));
        }
    }
    Ok(())
}
