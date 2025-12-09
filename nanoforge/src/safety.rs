use std::process;
use std::sync::Once;

static REGISTER_ONCE: Once = Once::new();

pub fn register_crash_handler() {
    REGISTER_ONCE.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as usize;
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut sa.sa_mask);

        if libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut()) != 0 {
            eprintln!("Failed to register SIGSEGV handler");
        }
        if libc::sigaction(libc::SIGILL, &sa, std::ptr::null_mut()) != 0 {
            eprintln!("Failed to register SIGILL handler");
        }
    });
}

extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    let addr = unsafe { (*info).si_addr() };
    eprintln!("\n\n!!! CRITICAL FAILURE !!!");
    eprintln!("Caught signal {}: Crash at address {:?}", sig, addr);
    eprintln!("This likely means the JIT-compiled code was invalid or memory was corrupted.");
    eprintln!("NanoForge is shutting down safely to prevent further damage.\n");

    // In a real system, we might try to longjmp out, but that's unsafe in Rust.
    // We just exit with a special code.
    process::exit(139); // Standard exit code for SIGSEGV
}
