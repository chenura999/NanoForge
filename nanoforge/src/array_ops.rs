//! High-Performance Array Operations
//!
//! Provides vectorized array operations using AVX2/AVX-512 when available.
//! These operations are designed for zero-copy interop with NumPy arrays.
//!
//! **Performance Optimizations**:
//! - JIT code compiled once and cached via OnceLock
//! - 4x loop unrolling (16 elements per iteration using 8 YMM registers)
//! - Aggressive prefetching (2 cache lines ahead)
//! - Non-temporal stores for large arrays (>1MB) to bypass cache

use crate::cpu_features::CpuFeatures;
use crate::jit_memory::DualMappedMemory;
use dynasmrt::{dynasm, x64::Assembler, DynasmApi, DynasmLabelApi};
use std::sync::OnceLock;

// Threshold for using non-temporal stores (elements)
// 1MB of i64 = 131072 elements
const NT_STORE_THRESHOLD: usize = 131072;

/// Cached JIT function for vec_add (regular stores)
struct CachedVecAdd {
    #[allow(dead_code)]
    memory: DualMappedMemory,
    func: extern "C" fn(*const i64, *const i64, *mut i64, usize),
}

unsafe impl Send for CachedVecAdd {}
unsafe impl Sync for CachedVecAdd {}

static VEC_ADD_AVX2: OnceLock<CachedVecAdd> = OnceLock::new();
static VEC_ADD_AVX2_NT: OnceLock<CachedVecAdd> = OnceLock::new();

/// Cached JIT function for vec_sum
struct CachedVecSum {
    #[allow(dead_code)]
    memory: DualMappedMemory,
    func: extern "C" fn(*const i64, usize) -> i64,
}

unsafe impl Send for CachedVecSum {}
unsafe impl Sync for CachedVecSum {}

static VEC_SUM_AVX2: OnceLock<CachedVecSum> = OnceLock::new();

/// Vector addition: C[i] = A[i] + B[i]
/// Uses AVX2 for 4x i64 parallelism when available
/// For arrays > 1MB with aligned output, uses non-temporal stores
pub fn vec_add_i64(a: &[i64], b: &[i64], c: &mut [i64]) {
    let n = a.len().min(b.len()).min(c.len());

    let features = CpuFeatures::detect();

    if features.has_avx2 && n >= 16 {
        // Check if output is 32-byte aligned for NT stores
        let c_aligned = (c.as_ptr() as usize) % 32 == 0;

        if n >= NT_STORE_THRESHOLD && c_aligned {
            // Large array with aligned output: use non-temporal stores
            let cached = VEC_ADD_AVX2_NT.get_or_init(|| {
                init_vec_add_avx2_nt().expect("Failed to initialize AVX2 NT vec_add")
            });
            (cached.func)(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
        } else {
            // Small/medium array or unaligned: use regular stores
            let cached = VEC_ADD_AVX2
                .get_or_init(|| init_vec_add_avx2().expect("Failed to initialize AVX2 vec_add"));
            (cached.func)(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
        }
    } else {
        // Scalar fallback
        for i in 0..n {
            c[i] = a[i] + b[i];
        }
    }
}

/// Initialize cached AVX2 vec_add function (regular stores)
fn init_vec_add_avx2() -> Result<CachedVecAdd, String> {
    let code = generate_vec_add_avx2_regular()?;

    let memory = DualMappedMemory::new(code.len().max(4096))
        .map_err(|e| format!("Failed to allocate JIT memory: {}", e))?;

    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), memory.rw_ptr, code.len());
    }
    memory.flush_icache();

    let func: extern "C" fn(*const i64, *const i64, *mut i64, usize) =
        unsafe { std::mem::transmute(memory.rx_ptr) };

    Ok(CachedVecAdd { memory, func })
}

/// Initialize cached AVX2 vec_add function with non-temporal stores
fn init_vec_add_avx2_nt() -> Result<CachedVecAdd, String> {
    let code = generate_vec_add_avx2_nt()?;

    let memory = DualMappedMemory::new(code.len().max(4096))
        .map_err(|e| format!("Failed to allocate JIT memory: {}", e))?;

    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), memory.rw_ptr, code.len());
    }
    memory.flush_icache();

    let func: extern "C" fn(*const i64, *const i64, *mut i64, usize) =
        unsafe { std::mem::transmute(memory.rx_ptr) };

    Ok(CachedVecAdd { memory, func })
}

/// Generate AVX2 vector add with regular stores
fn generate_vec_add_avx2_regular() -> Result<Vec<u8>, String> {
    let mut ops = Assembler::new().map_err(|e| e.to_string())?;

    dynasm!(ops
        ; .arch x64
        ; push rbx
        ; push r12
        ; push r13
        ; mov rbx, rcx
        ; mov r12, rdx
        ; mov r13, rdi

        ; xor rcx, rcx

        ; .align 32
        ; ->vec_loop_16:
        ; mov rax, rbx
        ; sub rax, rcx
        ; cmp rax, 16
        ; jl ->vec_loop_4

        ; prefetcht0 [r13 + rcx * 8 + 128]
        ; prefetcht0 [rsi + rcx * 8 + 128]

        ; vmovdqu ymm0, [r13 + rcx * 8]
        ; vmovdqu ymm1, [r13 + rcx * 8 + 32]
        ; vmovdqu ymm2, [r13 + rcx * 8 + 64]
        ; vmovdqu ymm3, [r13 + rcx * 8 + 96]

        ; vmovdqu ymm4, [rsi + rcx * 8]
        ; vmovdqu ymm5, [rsi + rcx * 8 + 32]
        ; vmovdqu ymm6, [rsi + rcx * 8 + 64]
        ; vmovdqu ymm7, [rsi + rcx * 8 + 96]

        ; vpaddq ymm0, ymm0, ymm4
        ; vpaddq ymm1, ymm1, ymm5
        ; vpaddq ymm2, ymm2, ymm6
        ; vpaddq ymm3, ymm3, ymm7

        ; vmovdqu [r12 + rcx * 8], ymm0
        ; vmovdqu [r12 + rcx * 8 + 32], ymm1
        ; vmovdqu [r12 + rcx * 8 + 64], ymm2
        ; vmovdqu [r12 + rcx * 8 + 96], ymm3

        ; add rcx, 16
        ; jmp ->vec_loop_16

        ; ->vec_loop_4:
        ; mov rax, rbx
        ; sub rax, rcx
        ; cmp rax, 4
        ; jl ->scalar_cleanup

        ; vmovdqu ymm0, [r13 + rcx * 8]
        ; vmovdqu ymm1, [rsi + rcx * 8]
        ; vpaddq ymm0, ymm0, ymm1
        ; vmovdqu [r12 + rcx * 8], ymm0

        ; add rcx, 4
        ; jmp ->vec_loop_4

        ; ->scalar_cleanup:
        ; cmp rcx, rbx
        ; jge ->done

        ; mov rax, [r13 + rcx * 8]
        ; add rax, [rsi + rcx * 8]
        ; mov [r12 + rcx * 8], rax
        ; inc rcx
        ; jmp ->scalar_cleanup

        ; ->done:
        ; pop r13
        ; pop r12
        ; pop rbx
        ; vzeroupper
        ; ret
    );

    let buf = ops.finalize().map_err(|e| format!("{:?}", e))?;
    Ok(buf.to_vec())
}

/// Generate AVX2 vector add with non-temporal stores
/// REQUIRES: Output buffer (rdx) must be 32-byte aligned
fn generate_vec_add_avx2_nt() -> Result<Vec<u8>, String> {
    let mut ops = Assembler::new().map_err(|e| e.to_string())?;

    dynasm!(ops
        ; .arch x64
        ; push rbx
        ; push r12
        ; push r13
        ; mov rbx, rcx          // rbx = n
        ; mov r12, rdx          // r12 = C (MUST be 32-byte aligned)
        ; mov r13, rdi          // r13 = A

        ; xor rcx, rcx          // rcx = i = 0

        // Main loop: 16 elements per iteration with NT stores
        ; .align 32
        ; ->vec_loop_16:
        ; mov rax, rbx
        ; sub rax, rcx
        ; cmp rax, 16
        ; jl ->vec_loop_4

        // Prefetch reads
        ; prefetcht0 [r13 + rcx * 8 + 128]
        ; prefetcht0 [rsi + rcx * 8 + 128]

        // Load 16 from A (unaligned OK)
        ; vmovdqu ymm0, [r13 + rcx * 8]
        ; vmovdqu ymm1, [r13 + rcx * 8 + 32]
        ; vmovdqu ymm2, [r13 + rcx * 8 + 64]
        ; vmovdqu ymm3, [r13 + rcx * 8 + 96]

        // Load 16 from B (unaligned OK)
        ; vmovdqu ymm4, [rsi + rcx * 8]
        ; vmovdqu ymm5, [rsi + rcx * 8 + 32]
        ; vmovdqu ymm6, [rsi + rcx * 8 + 64]
        ; vmovdqu ymm7, [rsi + rcx * 8 + 96]

        // Add
        ; vpaddq ymm0, ymm0, ymm4
        ; vpaddq ymm1, ymm1, ymm5
        ; vpaddq ymm2, ymm2, ymm6
        ; vpaddq ymm3, ymm3, ymm7

        // Non-temporal stores - output MUST be 32-byte aligned
        ; vmovntdq [r12 + rcx * 8], ymm0
        ; vmovntdq [r12 + rcx * 8 + 32], ymm1
        ; vmovntdq [r12 + rcx * 8 + 64], ymm2
        ; vmovntdq [r12 + rcx * 8 + 96], ymm3

        ; add rcx, 16
        ; jmp ->vec_loop_16

        // Secondary loop: 4 elements with NT stores
        ; ->vec_loop_4:
        ; mov rax, rbx
        ; sub rax, rcx
        ; cmp rax, 4
        ; jl ->scalar_cleanup

        ; vmovdqu ymm0, [r13 + rcx * 8]
        ; vmovdqu ymm1, [rsi + rcx * 8]
        ; vpaddq ymm0, ymm0, ymm1
        ; vmovntdq [r12 + rcx * 8], ymm0

        ; add rcx, 4
        ; jmp ->vec_loop_4

        // Scalar cleanup (regular stores for remainder)
        ; ->scalar_cleanup:
        ; cmp rcx, rbx
        ; jge ->done

        ; mov rax, [r13 + rcx * 8]
        ; add rax, [rsi + rcx * 8]
        ; mov [r12 + rcx * 8], rax
        ; inc rcx
        ; jmp ->scalar_cleanup

        ; ->done:
        ; sfence              // Ensure all NT stores complete before return
        ; pop r13
        ; pop r12
        ; pop rbx
        ; vzeroupper
        ; ret
    );

    let buf = ops.finalize().map_err(|e| format!("{:?}", e))?;
    Ok(buf.to_vec())
}

/// Vector sum: returns sum of all elements
pub fn vec_sum_i64(arr: &[i64]) -> i64 {
    let n = arr.len();

    let features = CpuFeatures::detect();

    if features.has_avx2 && n >= 16 {
        let cached = VEC_SUM_AVX2
            .get_or_init(|| init_vec_sum_avx2().expect("Failed to initialize AVX2 vec_sum"));
        (cached.func)(arr.as_ptr(), n)
    } else {
        arr.iter().sum()
    }
}

fn init_vec_sum_avx2() -> Result<CachedVecSum, String> {
    let code = generate_vec_sum_avx2_ultra()?;

    let memory = DualMappedMemory::new(code.len().max(4096))
        .map_err(|e| format!("Failed to allocate JIT memory: {}", e))?;

    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), memory.rw_ptr, code.len());
    }
    memory.flush_icache();

    let func: extern "C" fn(*const i64, usize) -> i64 =
        unsafe { std::mem::transmute(memory.rx_ptr) };

    Ok(CachedVecSum { memory, func })
}

fn generate_vec_sum_avx2_ultra() -> Result<Vec<u8>, String> {
    let mut ops = Assembler::new().map_err(|e| e.to_string())?;

    dynasm!(ops
        ; .arch x64
        ; vpxor ymm0, ymm0, ymm0
        ; vpxor ymm1, ymm1, ymm1
        ; vpxor ymm2, ymm2, ymm2
        ; vpxor ymm3, ymm3, ymm3

        ; xor rcx, rcx

        ; .align 32
        ; ->sum_loop_16:
        ; mov rax, rsi
        ; sub rax, rcx
        ; cmp rax, 16
        ; jl ->sum_loop_4

        ; prefetcht0 [rdi + rcx * 8 + 128]

        ; vmovdqu ymm4, [rdi + rcx * 8]
        ; vmovdqu ymm5, [rdi + rcx * 8 + 32]
        ; vmovdqu ymm6, [rdi + rcx * 8 + 64]
        ; vmovdqu ymm7, [rdi + rcx * 8 + 96]

        ; vpaddq ymm0, ymm0, ymm4
        ; vpaddq ymm1, ymm1, ymm5
        ; vpaddq ymm2, ymm2, ymm6
        ; vpaddq ymm3, ymm3, ymm7

        ; add rcx, 16
        ; jmp ->sum_loop_16

        ; ->sum_loop_4:
        ; mov rax, rsi
        ; sub rax, rcx
        ; cmp rax, 4
        ; jl ->sum_reduce

        ; vmovdqu ymm4, [rdi + rcx * 8]
        ; vpaddq ymm0, ymm0, ymm4

        ; add rcx, 4
        ; jmp ->sum_loop_4

        ; ->sum_reduce:
        ; vpaddq ymm0, ymm0, ymm1
        ; vpaddq ymm2, ymm2, ymm3
        ; vpaddq ymm0, ymm0, ymm2

        ; vextracti128 xmm1, ymm0, 1
        ; vpaddq xmm0, xmm0, xmm1
        ; vpsrldq xmm1, xmm0, 8
        ; vpaddq xmm0, xmm0, xmm1
        ; vmovq rax, xmm0

        ; ->scalar_loop:
        ; cmp rcx, rsi
        ; jge ->sum_done
        ; add rax, [rdi + rcx * 8]
        ; inc rcx
        ; jmp ->scalar_loop

        ; ->sum_done:
        ; vzeroupper
        ; ret
    );

    let buf = ops.finalize().map_err(|e| format!("{:?}", e))?;
    Ok(buf.to_vec())
}

/// In-place scale: arr[i] *= scalar
pub fn vec_scale_i64(arr: &mut [i64], scalar: i64) {
    for x in arr.iter_mut() {
        *x *= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_add_basic() {
        let a = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![10i64, 20, 30, 40, 50, 60, 70, 80];
        let mut c = vec![0i64; 8];

        vec_add_i64(&a, &b, &mut c);

        let expected: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(c, expected);
    }

    #[test]
    fn test_vec_add_large() {
        let n = 100_000;
        let a: Vec<i64> = (0..n).collect();
        let b: Vec<i64> = (0..n).map(|x| x * 2).collect();
        let mut c = vec![0i64; n as usize];

        vec_add_i64(&a, &b, &mut c);

        for i in 0..n as usize {
            assert_eq!(c[i], a[i] + b[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_vec_add_nt_threshold() {
        // Test at NT threshold boundary (>131072 elements = 1MB)
        // Note: NT stores require 32-byte alignment which Vec may not guarantee
        // This test will use regular stores if unaligned
        let n = 200_000;
        let a: Vec<i64> = (0..n).collect();
        let b: Vec<i64> = (0..n).map(|x| x * 2).collect();
        let mut c = vec![0i64; n as usize];

        vec_add_i64(&a, &b, &mut c);

        // Spot check
        assert_eq!(c[0], 0);
        assert_eq!(c[1000], 1000 + 2000);
        assert_eq!(c[n as usize - 1], (n - 1) as i64 + (n - 1) as i64 * 2);
    }

    #[test]
    fn test_vec_add_misaligned() {
        let a = vec![
            1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        let b = vec![
            10i64, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
        ];
        let mut c = vec![0i64; 19];

        vec_add_i64(&a, &b, &mut c);

        let expected: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(c, expected);
    }

    #[test]
    fn test_vec_sum() {
        let arr: Vec<i64> = (1..=100).collect();
        let result = vec_sum_i64(&arr);
        let expected: i64 = (1..=100).sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vec_sum_large() {
        let n = 100_000i64;
        let arr: Vec<i64> = (0..n).collect();
        let result = vec_sum_i64(&arr);
        let expected: i64 = (0..n).sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vec_scale() {
        let mut arr = vec![1i64, 2, 3, 4, 5];
        vec_scale_i64(&mut arr, 10);
        assert_eq!(arr, vec![10, 20, 30, 40, 50]);
    }
}
