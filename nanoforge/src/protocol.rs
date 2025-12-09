#[derive(Debug, PartialEq)]
pub enum Command {
    Register(i32),
    Read,
    Error(String),
}

pub fn parse_command(line: &str) -> Command {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.is_empty() {
        return Command::Error("Empty command".to_string());
    }

    match parts[0] {
        "REGISTER" => {
            if parts.len() < 2 {
                return Command::Error("Missing PID".to_string());
            }
            match parts[1].parse::<i32>() {
                Ok(pid) => Command::Register(pid),
                Err(_) => Command::Error("Invalid PID".to_string()),
            }
        }
        "READ" => Command::Read,
        _ => Command::Error("Unknown Command".to_string()),
    }
}
