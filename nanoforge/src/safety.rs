use std::ptr;
use std::sync::Once;

// Global jump buffer pointer (Simpler and signal-safe for PoC than RefCell)
// WARNING: Not thread-safe! Only one thread can use the sandbox at a time with this.
static mut GLOBAL_JMP_BUF: *mut libc::c_void = ptr::null_mut();

static INIT: Once = Once::new();

// Manual FFI for setjmp/longjmp since libc doesn't always expose them cleanly
extern "C" {
    #[link_name = "setjmp"]
    fn setjmp(env: *mut libc::c_void) -> i32;
    #[link_name = "longjmp"]
    fn longjmp(env: *mut libc::c_void, val: i32);
}

unsafe extern "C" fn signal_handler(_sig: i32) {
    // Recover the jump buffer
    if !GLOBAL_JMP_BUF.is_null() {
        longjmp(GLOBAL_JMP_BUF, 1);
    }
    // If no buffer, we crash normally
    libc::abort();
}

pub fn install_signal_handler() {
    INIT.call_once(|| unsafe {
        let sa = libc::sigaction {
            sa_sigaction: signal_handler as usize,
            sa_flags: libc::SA_NODEFER,
            sa_mask: std::mem::zeroed(),
            #[cfg(target_os = "linux")]
            sa_restorer: None,
        };
        libc::sigaction(libc::SIGILL, &sa, ptr::null_mut());
        libc::sigaction(libc::SIGSEGV, &sa, ptr::null_mut());
    });
}

/// Runs the closure. If a signal (SIGILL/SIGSEGV) occurs, returns Err.
pub fn run_safely<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce() -> R,
{
    unsafe {
        // Allocate a jmp_buf. On x86_64 glibc, it's 200 bytes.
        // We'll use a generous buffer and cast it.
        let mut jb = [0u8; 512];

        // Save it in Global
        GLOBAL_JMP_BUF = jb.as_mut_ptr() as *mut libc::c_void;

        // setjmp returns 0 on direct call, non-zero on longjmp
        if setjmp(jb.as_mut_ptr() as *mut libc::c_void) == 0 {
            let result = f();
            // Clear Global
            GLOBAL_JMP_BUF = ptr::null_mut();
            Ok(result)
        } else {
            // We came from the signal handler
            GLOBAL_JMP_BUF = ptr::null_mut();
            Err("Caught fatal signal (SIGILL/SIGSEGV)".to_string())
        }
    }
}
