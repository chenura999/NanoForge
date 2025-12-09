use clap::Parser;
use nanoforge::profiler::Profiler;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::thread;
use tracing::{error, info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the Unix Domain Socket
    #[arg(short, long, default_value = "/tmp/nanoforge.sock")]
    socket_path: String,
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("NanoForge Daemon starting...");

    if Path::new(&args.socket_path).exists() {
        if let Err(e) = fs::remove_file(&args.socket_path) {
            error!("Failed to remove existing socket: {}", e);
            return;
        }
    }

    let listener = match UnixListener::bind(&args.socket_path) {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind to socket {}: {}", args.socket_path, e);
            return;
        }
    };

    info!("Listening on {}", args.socket_path);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| handle_client(stream));
            }
            Err(err) => {
                error!("Error accepting connection: {}", err);
            }
        }
    }
}

fn handle_client(mut stream: UnixStream) {
    let stream_clone = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to clone stream: {}", e);
            return;
        }
    };
    let mut reader = BufReader::new(stream_clone);
    let mut profiler: Option<Profiler> = None;

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let line = line.trim();
                let parts: Vec<&str> = line.split_whitespace().collect();

                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "REGISTER" => {
                        if parts.len() < 2 {
                            let _ = stream.write_all(b"ERROR Missing PID\n");
                            continue;
                        }
                        if let Ok(pid) = parts[1].parse::<i32>() {
                            // SECURITY CHECK: Verify Client UID == Target PID Owner
                            match check_permissions(&stream, pid) {
                                Ok(_) => {
                                    info!("Security Check Passed for PID: {}", pid);
                                }
                                Err(e) => {
                                    warn!("Security Check Failed: {}", e);
                                    let msg = format!("ERROR Security: {}\n", e);
                                    let _ = stream.write_all(msg.as_bytes());
                                    continue;
                                }
                            }

                            info!("Registering PID: {}", pid);
                            match Profiler::new_instruction_counter(pid) {
                                Ok(p) => {
                                    p.enable(); // Start profiling immediately
                                    profiler = Some(p);
                                    let _ = stream.write_all(b"OK\n");
                                }
                                Err(e) => {
                                    error!("Failed to create profiler for PID {}: {}", pid, e);
                                    let msg = format!("ERROR {}\n", e);
                                    let _ = stream.write_all(msg.as_bytes());
                                }
                            }
                        } else {
                            let _ = stream.write_all(b"ERROR Invalid PID\n");
                        }
                    }
                    "READ" => {
                        if let Some(ref p) = profiler {
                            let count = p.read();
                            let response = format!("{}\n", count);
                            let _ = stream.write_all(response.as_bytes());
                        } else {
                            let _ = stream.write_all(b"ERROR Not Registered\n");
                        }
                    }
                    _ => {
                        warn!("Unknown command received: {}", parts[0]);
                        let _ = stream.write_all(b"ERROR Unknown Command\n");
                    }
                }
            }
            Err(e) => {
                error!("Error reading from socket: {}", e);
                break;
            }
        }
    }
}

fn check_permissions(stream: &UnixStream, target_pid: i32) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::io::AsRawFd;

    // 1. Get Client UID via libc::getsockopt
    let fd = stream.as_raw_fd();
    let client_uid = unsafe {
        let mut ucred = libc::ucred {
            pid: 0,
            uid: 0,
            gid: 0,
        };
        let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
        if libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut ucred as *mut _ as *mut libc::c_void,
            &mut len,
        ) == 0
        {
            ucred.uid
        } else {
            return Err("Failed to get peer credentials".to_string());
        }
    };

    // 2. Get Target PID Owner
    let proc_path = format!("/proc/{}", target_pid);
    let metadata =
        fs::metadata(&proc_path).map_err(|e| format!("Failed to stat {}: {}", proc_path, e))?;
    let target_uid = metadata.uid();

    // 3. Compare
    if client_uid != target_uid {
        return Err(format!(
            "Permission Denied: Client UID {} cannot profile Target UID {}",
            client_uid, target_uid
        ));
    }

    Ok(())
}
