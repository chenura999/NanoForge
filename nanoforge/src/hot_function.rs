#![allow(dead_code)]
use crate::jit_memory::DualMappedMemory;
use crossbeam::epoch::{self, Atomic, Owned};
use std::sync::atomic::Ordering;

// A wrapper around the raw function pointer that we can manage with EBR
pub struct JittedCode {
    // We keep memory here to ensure it stays alive as long as the code is used
    pub _memory: DualMappedMemory,
    pub func_ptr: extern "C" fn(u64) -> u64,
}

// SAFETY: JittedCode is immutable once created.
unsafe impl Send for JittedCode {}
unsafe impl Sync for JittedCode {}

pub struct HotFunction {
    // The active implementation.
    // We use crossbeam::epoch::Atomic to manage the lifetime of the pointer.
    current: Atomic<JittedCode>,
}

impl HotFunction {
    pub fn new(initial_code: DualMappedMemory, offset: usize) -> Self {
        let func_ptr: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(initial_code.rx_ptr.add(offset)) };

        let code = JittedCode {
            _memory: initial_code,
            func_ptr,
        };

        Self {
            current: Atomic::new(code),
        }
    }

    pub fn call(&self, arg: u64) -> u64 {
        // 1. Enter critical section (pin the epoch)
        let guard = epoch::pin();

        // 2. Load the current implementation
        let shared = self.current.load(Ordering::Acquire, &guard);

        // 3. Execute
        // Safety: The guard ensures 'shared' remains valid during this call.
        // We must unwrap because we initialized it.
        let code = unsafe { shared.as_ref() }.expect("HotFunction is null!");
        (code.func_ptr)(arg)
    }

    pub fn update(&self, new_memory: DualMappedMemory, offset: usize) {
        let func_ptr: extern "C" fn(u64) -> u64 =
            unsafe { std::mem::transmute(new_memory.rx_ptr.add(offset)) };

        let new_code = JittedCode {
            _memory: new_memory,
            func_ptr,
        };

        // 1. Enter critical section
        let guard = epoch::pin();

        // 2. Atomic Swap
        // We move 'new_code' into an Owned pointer, then swap it into the Atomic.
        let old = self
            .current
            .swap(Owned::new(new_code), Ordering::Release, &guard);

        // 3. Defer Destruction
        // 'old' is a Shared pointer to the previous JittedCode.
        // We need to schedule its destruction only after all threads have left the previous epoch.
        unsafe {
            // We need to take ownership of the Shared pointer to drop it.
            // crossbeam's defer_destroy handles this automatically if we use 'defer'.
            // But here we want to explicitly drop the DualMappedMemory when it's safe.
            guard.defer_destroy(old);
        }

        println!("HotFunction: Swapped implementation. Old memory will be freed safely.");
    }
}
