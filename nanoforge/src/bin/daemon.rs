use nanoforge::profiler::Profiler;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

// We need to import Profiler from the library.
// Since `nanoforge` is a binary crate, we might need to move `profiler.rs` to a lib.rs
// or just copy the module declaration if we can't easily refactor to a workspace.
// For now, let's assume we can declare the module here or we need to refactor `main.rs` to `lib.rs`.
// Actually, `src/bin/daemon.rs` is a separate binary in the same crate.
// It can't easily access modules defined in `src/main.rs`.
// We should move the modules to `src/lib.rs` to share them.

// TEMPORARY: We will duplicate the Profiler struct here for the daemon
// OR we will refactor the project structure.
// Refactoring is better.

fn main() {
    println!("NanoForge Daemon starting...");

    let socket_path = "/tmp/nanoforge.sock";
    if Path::new(socket_path).exists() {
        fs::remove_file(socket_path).unwrap();
    }

    let listener = UnixListener::bind(socket_path).unwrap();
    println!("Listening on {}", socket_path);

    // Map PID -> Profiler
    // We use a Mutex to share state across threads (if we were multi-threaded per connection)
    // For simplicity, we can handle connections sequentially or spawn a thread per connection.
    // Since a profiler instance is tied to a connection session usually (or we can keep a global map).
    // Let's keep it simple: One connection = One session.
    // The client registers a PID, and we keep the profiler for that connection.

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                thread::spawn(|| handle_client(stream));
            }
            Err(err) => {
                println!("Error accepting connection: {}", err);
            }
        }
    }
}

fn handle_client(mut stream: UnixStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
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
                            println!("Registering PID: {}", pid);
                            match Profiler::new_instruction_counter(pid) {
                                Ok(p) => {
                                    p.enable(); // Start profiling immediately
                                    profiler = Some(p);
                                    let _ = stream.write_all(b"OK\n");
                                }
                                Err(e) => {
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
                        let _ = stream.write_all(b"ERROR Unknown Command\n");
                    }
                }
            }
            Err(e) => {
                println!("Error reading from socket: {}", e);
                break;
            }
        }
    }
}
