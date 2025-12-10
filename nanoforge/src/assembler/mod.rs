// Re-export the appropriate CodeGenerator based on the architecture.

#[cfg(target_arch = "x86_64")]
pub mod x64;
#[cfg(target_arch = "x86_64")]
pub use self::x64::CodeGenerator;
#[cfg(target_arch = "x86_64")]
pub use self::x64::JitBuilder;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use self::aarch64::CodeGenerator;
#[cfg(target_arch = "aarch64")]
pub use self::aarch64::JitBuilder;

// If neither, we might want to fail or provide a stub.
// For now, we assume one of the two.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("Nanoforge only supports x86_64 and aarch64");
pub mod manual_test;
