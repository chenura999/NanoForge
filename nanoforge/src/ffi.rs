//! Foreign Function Interface (FFI) for NanoForge
//!
//! Provides C-compatible API for calling NanoForge from other languages.
//! Use cbindgen to generate the C header file.

use crate::ai_optimizer::{ContextualBandit, OptimizationFeatures};
use crate::cpu_features::CpuFeatures;
use crate::parser::Parser;
use crate::variant_generator::VariantGenerator;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

/// Opaque handle to a compiled function
#[repr(C)]
pub struct NanoFunction {
    func_ptr: extern "C" fn(u64) -> u64,
}

/// Opaque handle to the AI optimizer
#[repr(C)]
pub struct NanoOptimizer {
    bandit: Box<ContextualBandit>,
    variant_names: Vec<String>,
}

/// Result codes for FFI functions
#[repr(C)]
pub enum NanoResult {
    Ok = 0,
    ErrorParseFailed = 1,
    ErrorCompileFailed = 2,
    ErrorNullPointer = 3,
    ErrorInvalidUtf8 = 4,
    ErrorIoFailed = 5,
}

// ============================================================================
// FFI Functions
// ============================================================================

/// Initialize NanoForge and detect CPU features
/// Returns a string with detected features (caller must free with nanoforge_free_string)
#[no_mangle]
pub extern "C" fn nanoforge_init() -> *mut c_char {
    let features = CpuFeatures::detect();
    let summary = features.summary();
    match CString::new(summary) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by NanoForge
#[no_mangle]
pub extern "C" fn nanoforge_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Compile a NanoForge script and return the best function
/// Returns null on failure
#[no_mangle]
pub extern "C" fn nanoforge_compile(source: *const c_char) -> *mut NanoFunction {
    if source.is_null() {
        return ptr::null_mut();
    }

    let source_str = match unsafe { CStr::from_ptr(source) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let mut parser = Parser::new();
    let program = match parser.parse(source_str) {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    let generator = VariantGenerator::new();
    let variants = match generator.generate_variants(&program) {
        Ok(v) => v,
        Err(_) => return ptr::null_mut(),
    };

    // Return the first (scalar) variant for simplicity
    if variants.is_empty() {
        return ptr::null_mut();
    }

    let func = Box::new(NanoFunction {
        func_ptr: variants[0].func_ptr,
    });

    Box::into_raw(func)
}

/// Execute a compiled function
#[no_mangle]
pub extern "C" fn nanoforge_execute(func: *const NanoFunction, input: u64) -> u64 {
    if func.is_null() {
        return 0;
    }
    let f = unsafe { &*func };
    (f.func_ptr)(input)
}

/// Free a compiled function
#[no_mangle]
pub extern "C" fn nanoforge_free_function(func: *mut NanoFunction) {
    if !func.is_null() {
        unsafe {
            let _ = Box::from_raw(func);
        }
    }
}

/// Create a new AI optimizer
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_new() -> *mut NanoOptimizer {
    let variant_names = vec![
        "Scalarx1".to_string(),
        "Scalarx2".to_string(),
        "AVX2x2".to_string(),
        "AVX2x4".to_string(),
    ];
    let bandit = ContextualBandit::new(variant_names.clone());

    let opt = Box::new(NanoOptimizer {
        bandit: Box::new(bandit),
        variant_names,
    });

    Box::into_raw(opt)
}

/// Select variant using AI optimizer
/// Returns the index of the selected variant
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_select(opt: *mut NanoOptimizer, input_size: u64) -> i32 {
    if opt.is_null() {
        return -1;
    }
    let optimizer = unsafe { &mut *opt };
    let features = OptimizationFeatures::new(input_size);
    optimizer.bandit.select(&features) as i32
}

/// Update AI optimizer with feedback
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_update(
    opt: *mut NanoOptimizer,
    input_size: u64,
    variant_idx: i32,
    cycles: u64,
    best_cycles: u64,
) {
    if opt.is_null() || variant_idx < 0 {
        return;
    }
    let optimizer = unsafe { &mut *opt };
    let features = OptimizationFeatures::new(input_size);
    optimizer
        .bandit
        .update_with_performance(&features, variant_idx as usize, cycles, best_cycles);
}

/// Save AI optimizer to file
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_save(
    opt: *const NanoOptimizer,
    path: *const c_char,
) -> NanoResult {
    if opt.is_null() || path.is_null() {
        return NanoResult::ErrorNullPointer;
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return NanoResult::ErrorInvalidUtf8,
    };

    let optimizer = unsafe { &*opt };
    match optimizer.bandit.save_to_file(Path::new(path_str)) {
        Ok(_) => NanoResult::Ok,
        Err(_) => NanoResult::ErrorIoFailed,
    }
}

/// Load AI optimizer from file
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_load(path: *const c_char) -> *mut NanoOptimizer {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let variant_names = vec![
        "Scalarx1".to_string(),
        "Scalarx2".to_string(),
        "AVX2x2".to_string(),
        "AVX2x4".to_string(),
    ];

    let bandit = ContextualBandit::load_or_new(Path::new(path_str), variant_names.clone());

    let opt = Box::new(NanoOptimizer {
        bandit: Box::new(bandit),
        variant_names,
    });

    Box::into_raw(opt)
}

/// Free AI optimizer
#[no_mangle]
pub extern "C" fn nanoforge_optimizer_free(opt: *mut NanoOptimizer) {
    if !opt.is_null() {
        unsafe {
            let _ = Box::from_raw(opt);
        }
    }
}

/// Get version string
#[no_mangle]
pub extern "C" fn nanoforge_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
