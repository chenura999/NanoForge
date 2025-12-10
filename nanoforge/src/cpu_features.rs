//! CPU Feature Detection using CPUID
//!
//! Detects available ISA extensions at runtime to generate appropriate variants.

use std::arch::x86_64::__cpuid;

/// Detected CPU features for variant generation
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub has_sse2: bool,
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512vl: bool,
    pub has_avx512bw: bool,
    pub has_amx_bf16: bool,
    pub has_amx_int8: bool,
    pub has_amx_tile: bool,
}

impl CpuFeatures {
    /// Detect CPU features using CPUID instruction
    pub fn detect() -> Self {
        let mut features = CpuFeatures::default();

        unsafe {
            // Basic feature flags (CPUID EAX=1)
            let cpuid1 = __cpuid(1);
            features.has_sse2 = (cpuid1.edx & (1 << 26)) != 0;
            features.has_sse4_1 = (cpuid1.ecx & (1 << 19)) != 0;
            features.has_sse4_2 = (cpuid1.ecx & (1 << 20)) != 0;
            features.has_avx = (cpuid1.ecx & (1 << 28)) != 0;

            // Extended feature flags (CPUID EAX=7, ECX=0)
            let cpuid7 = __cpuid(7);
            features.has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;
            features.has_avx512f = (cpuid7.ebx & (1 << 16)) != 0;
            features.has_avx512vl = (cpuid7.ebx & (1 << 31)) != 0;
            features.has_avx512bw = (cpuid7.ebx & (1 << 30)) != 0;

            // AMX features (CPUID EAX=7, ECX=0, EDX bits)
            features.has_amx_bf16 = (cpuid7.edx & (1 << 22)) != 0;
            features.has_amx_int8 = (cpuid7.edx & (1 << 25)) != 0;
            features.has_amx_tile = (cpuid7.edx & (1 << 24)) != 0;
        }

        features
    }

    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.has_avx2
    }

    /// Check if AVX-512 foundation is available
    pub fn has_avx512(&self) -> bool {
        self.has_avx512f
    }

    /// Check if AMX (Advanced Matrix Extensions) is available
    pub fn has_amx(&self) -> bool {
        self.has_amx_tile && (self.has_amx_bf16 || self.has_amx_int8)
    }

    /// Get a summary of detected features
    pub fn summary(&self) -> String {
        let mut features = vec![];
        if self.has_sse2 {
            features.push("SSE2");
        }
        if self.has_sse4_2 {
            features.push("SSE4.2");
        }
        if self.has_avx {
            features.push("AVX");
        }
        if self.has_avx2 {
            features.push("AVX2");
        }
        if self.has_avx512f {
            features.push("AVX-512F");
        }
        if self.has_amx_tile {
            features.push("AMX");
        }
        features.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let features = CpuFeatures::detect();
        println!("Detected CPU features: {}", features.summary());
        // At minimum, SSE2 should be available on any x86_64
        assert!(features.has_sse2);
    }
}
