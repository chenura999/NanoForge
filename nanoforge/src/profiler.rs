use libc::{c_int, c_long, c_void};
use std::io::Error;
use std::mem;

// Bindings for perf_event_open

// Re-defining the struct to match Linux kernel layout (u32, u32, u64...)
// This is risky without a crate, but "Low-Level Optimization Engineer" implies we can do this.
#[repr(C)]
#[derive(Default)]
pub struct PerfEventAttr {
    pub type_: u32,
    pub size: u32,
    pub config: u64,
    pub sample_period: u64,
    pub sample_type: u64,
    pub read_format: u64,
    pub flags: u64, // disabled, inherit, pinned, etc. (bitfields)
    pub wakeup_events: u32,
    pub bp_type: u32,
    pub bp_addr: u64,
    pub bp_len: u64,
    pub branch_sample_type: u64,
    pub sample_regs_user: u64,
    pub sample_stack_user: u32,
    pub clockid: i32,
    pub sample_regs_intr: u64,
    pub aux_watermark: u32,
    pub sample_max_stack: u16,
    pub __reserved_2: u16,
}

const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;

extern "C" {
    fn syscall(number: c_long, ...) -> c_long;
}

const SYS_PERF_EVENT_OPEN: c_long = 298; // x86_64

pub struct Profiler {
    fd: c_int,
}

impl Profiler {
    pub fn new_instruction_counter(pid: i32) -> Result<Self, String> {
        Self::new(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, pid)
    }

    fn new(type_: u32, config: u64, pid: i32) -> Result<Self, String> {
        let mut attr: PerfEventAttr = unsafe { mem::zeroed() };
        attr.type_ = type_;
        attr.size = mem::size_of::<PerfEventAttr>() as u32;
        attr.config = config;
        attr.flags = 1; // disabled = 1 (start disabled)
                        // We also want exclude_kernel = 1, exclude_hv = 1.
                        // Bit 1: disabled
                        // Bit 2: inherit
                        // Bit 3: pinned
                        // Bit 4: exclusive
                        // Bit 5: exclude_user
                        // Bit 6: exclude_kernel
                        // Bit 7: exclude_hv
                        // ...
                        // 1 | (1 << 5) | (1 << 6) is messy.
                        // Let's just set disabled=1 for now.

        // pid = 0 (current process), cpu = -1 (any cpu), group_fd = -1, flags = 0
        let fd = unsafe {
            syscall(
                SYS_PERF_EVENT_OPEN,
                &attr as *const PerfEventAttr,
                pid,
                -1,
                -1,
                0,
            )
        };

        if fd < 0 {
            return Err(format!(
                "perf_event_open failed: {}",
                Error::last_os_error()
            ));
        }

        Ok(Profiler { fd: fd as c_int })
    }

    pub fn enable(&self) {
        // PERF_EVENT_IOC_ENABLE is 0x2400
        const PERF_EVENT_IOC_ENABLE: c_long = 0x2400;
        unsafe { libc::ioctl(self.fd, PERF_EVENT_IOC_ENABLE as _, 0) };
    }

    pub fn disable(&self) {
        const PERF_EVENT_IOC_DISABLE: c_long = 0x2401;
        unsafe { libc::ioctl(self.fd, PERF_EVENT_IOC_DISABLE as _, 0) };
    }

    pub fn read(&self) -> u64 {
        let mut count: u64 = 0;
        let ret = unsafe {
            libc::read(
                self.fd,
                &mut count as *mut _ as *mut c_void,
                mem::size_of::<u64>(),
            )
        };
        if ret != mem::size_of::<u64>() as isize {
            return 0;
        }
        count
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

pub trait ProfileSource: Send + Sync {
    fn read(&self) -> u64;
    fn enable(&self);
    fn disable(&self);
}

impl ProfileSource for Profiler {
    fn read(&self) -> u64 {
        self.read()
    }
    fn enable(&self) {
        self.enable()
    }
    fn disable(&self) {
        self.disable()
    }
}

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

pub struct RemoteProfiler {
    stream: Mutex<UnixStream>,
}

impl RemoteProfiler {
    pub fn new(pid: i32) -> Result<Self, String> {
        let socket_path = "/tmp/nanoforge.sock";
        let mut stream = UnixStream::connect(socket_path).map_err(|e| e.to_string())?;

        // Register
        let cmd = format!("REGISTER {}\n", pid);
        stream
            .write_all(cmd.as_bytes())
            .map_err(|e| e.to_string())?;

        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut response = String::new();
        reader.read_line(&mut response).map_err(|e| e.to_string())?;

        if response.trim() != "OK" {
            return Err(format!("Daemon registration failed: {}", response.trim()));
        }

        Ok(RemoteProfiler {
            stream: Mutex::new(stream),
        })
    }
}

use std::sync::Mutex;

impl ProfileSource for RemoteProfiler {
    fn read(&self) -> u64 {
        let mut stream = self.stream.lock().unwrap();
        if stream.write_all(b"READ\n").is_err() {
            return 0;
        }

        let stream_clone = match stream.try_clone() {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let mut reader = BufReader::new(stream_clone);
        let mut response = String::new();
        if reader.read_line(&mut response).is_err() {
            return 0;
        }

        response.trim().parse().unwrap_or(0)
    }

    fn enable(&self) {
        // Daemon enables automatically on register
    }

    fn disable(&self) {
        // Daemon cleans up on connection close
    }
}
