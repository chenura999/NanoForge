use crate::ir::{Function, Instruction, Opcode, Operand, Program};
use std::collections::HashMap;

pub struct Parser {
    tokens: Vec<String>,
    pos: usize,
    symbol_table: HashMap<String, u8>, // Per-function symbol table
    next_reg: u8,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            pos: 0,
            symbol_table: HashMap::new(),
            next_reg: 1,
        }
    }

    fn tokenize(source: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = source.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                i += 1;
            } else if "(){},=+-".contains(c) {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                // Check for ==, !=, <=, >=
                if i + 1 < chars.len() {
                    let next = chars[i + 1];
                    if (c == '=' || c == '!' || c == '<' || c == '>') && next == '=' {
                        tokens.push(format!("{}{}", c, next));
                        i += 2;
                        continue;
                    }
                }
                tokens.push(c.to_string());
                i += 1;
            } else {
                current.push(c);
                i += 1;
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    fn peek(&self) -> Option<&String> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &str) -> Result<(), String> {
        let t = self.consume().ok_or("Unexpected end of input")?;
        if t == expected {
            Ok(())
        } else {
            Err(format!("Expected '{}', found '{}'", expected, t))
        }
    }

    fn get_or_alloc_reg(&mut self, name: &str) -> u8 {
        if let Some(&reg) = self.symbol_table.get(name) {
            reg
        } else {
            let reg = self.next_reg;
            self.next_reg += 1;
            self.symbol_table.insert(name.to_string(), reg);
            reg
        }
    }

    fn parse_operand(&mut self, token: &str) -> Operand {
        if let Ok(num) = token.parse::<i32>() {
            Operand::Imm(num)
        } else {
            let reg = self.get_or_alloc_reg(token);
            Operand::Reg(reg)
        }
    }

    pub fn parse(&mut self, source: &str) -> Result<Program, String> {
        self.tokens = Self::tokenize(source);
        self.pos = 0;
        let mut program = Program::new();

        while self.peek().is_some() {
            if self.peek().unwrap() == "fn" {
                program.add_function(self.parse_function()?);
            } else {
                // Legacy mode: Wrap top-level code in "main"
                // Reset pos? No, just parse body as main.
                // But we must assume the whole file is the main body if it doesn't start with fn.
                // If we mixed fn and non-fn, that's ambiguous.
                // Requirement: implicit main only if NO explicit fns BEFORE it?
                // Let's simplified: If tok != fn, parse 'main' until end or another fn?
                // Currently assume whole file is main if not starting with fn.
                if !program.functions.is_empty() {
                    return Err("Cannot mix top-level code with functions".to_string());
                }
                program.add_function(self.parse_implicit_main()?);
                break;
            }
        }
        Ok(program)
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        self.expect("fn")?;
        let name = self.consume().ok_or("Expected function name")?;
        self.expect("(")?;

        let mut args = Vec::new();
        while let Some(t) = self.peek() {
            if t == ")" {
                break;
            }
            if t == "," {
                self.consume();
                continue;
            }
            args.push(self.consume().unwrap());
        }
        self.expect(")")?;
        self.expect("{")?;

        // Reset symbol table for new function
        self.symbol_table.clear();
        self.next_reg = 1; // 0 is Ret

        let mut func = Function::new(&name, args.clone());

        for (i, arg) in args.iter().enumerate() {
            let dest_reg = self.get_or_alloc_reg(arg);
            // Assign arguments to registers using Stack Loading
            // Arguments are pushed by caller.
            // Stack: [Old RBP] [Ret Addr] [Arg 0] [Arg 1] ...
            // Arg 0 is at RBP + 16.
            // We use LoadArg opcode which Compiler translates to mov reg, [rbp + offset]
            func.push(Instruction {
                op: Opcode::LoadArg(i),
                dest: Some(Operand::Reg(dest_reg)),
                src1: None,
                src2: None,
            });
        }

        while let Some(t) = self.peek() {
            if t == "}" {
                self.consume();
                return Ok(func);
            }
            self.parse_statement(&mut func)?;
        }
        Err("Unexpected end of function".to_string())
    }

    fn parse_implicit_main(&mut self) -> Result<Function, String> {
        self.symbol_table.clear();
        self.next_reg = 1;
        let mut func = Function::new("main", vec![]);
        while self.peek().is_some() {
            self.parse_statement(&mut func)?;
        }
        Ok(func)
    }

    fn parse_statement(&mut self, func: &mut Function) -> Result<(), String> {
        let t = self.consume().ok_or("Unexpected EOF")?;

        match t.as_str() {
            "return" => {
                let val_token = self.consume().ok_or("Expected return value")?;
                let val = self.parse_operand(&val_token);
                func.push(Instruction {
                    op: Opcode::Mov,
                    dest: Some(Operand::Reg(0)),
                    src1: Some(val),
                    src2: None,
                });
                func.push(Instruction {
                    op: Opcode::Ret,
                    dest: None,
                    src1: None,
                    src2: None,
                });
            }
            "label" => {
                let name = self.consume().ok_or("Expected label name")?;
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(name)),
                    src1: None,
                    src2: None,
                });
            }
            "goto" => {
                let name = self.consume().ok_or("Expected goto label")?;
                func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(name)),
                    src1: None,
                    src2: None,
                });
            }
            "if" => {
                // if x goto L
                // if x == y goto L
                let lhs_token = self.consume().ok_or("Expected if condition")?;
                let next = self.consume().ok_or("Expected if op or goto")?;

                if next == "goto" {
                    // if x goto L
                    let label = self.consume().ok_or("Expected label")?;
                    let lhs = self.parse_operand(&lhs_token);
                    func.push(Instruction {
                        op: Opcode::Jnz,
                        dest: Some(Operand::Label(label)),
                        src1: Some(lhs),
                        src2: None,
                    });
                } else {
                    // comparison
                    let op_str = next;
                    let rhs_token = self.consume().ok_or("Expected rhs")?;
                    let goto_kw = self.consume().ok_or("Expected goto")?;
                    if goto_kw != "goto" {
                        return Err("Expected goto".to_string());
                    }
                    let label = self.consume().ok_or("Expected label")?;

                    let lhs = self.parse_operand(&lhs_token);
                    let rhs = self.parse_operand(&rhs_token);

                    func.push(Instruction {
                        op: Opcode::Cmp,
                        dest: None,
                        src1: Some(lhs),
                        src2: Some(rhs),
                    });

                    let jump_op = match op_str.as_str() {
                        "==" => Opcode::Je,
                        "!=" => Opcode::Jne,
                        "<" => Opcode::Jl,
                        "<=" => Opcode::Jle,
                        ">" => Opcode::Jg,
                        ">=" => Opcode::Jge,
                        _ => return Err(format!("Unknown op {}", op_str)),
                    };
                    func.push(Instruction {
                        op: jump_op,
                        dest: Some(Operand::Label(label)),
                        src1: None,
                        src2: None,
                    });
                }
            }
            _ => {
                // Assignment `x = ...`
                let dest_name = t;
                let eq = self.consume().ok_or("Expected =")?;
                if eq != "=" {
                    return Err(format!("Expected =, found {}", eq));
                }

                // RHS: `val` or `val + val` or `func(arg...)`
                let token1 = self.consume().ok_or("Expected RHS")?;
                // Check if it's a call
                // Peek next. If `(`, it's a call.
                if self.peek() == Some(&"(".to_string()) {
                    // Call: name ( args )
                    let func_name = token1;
                    self.consume(); // (
                    let mut args = Vec::new();
                    while let Some(at) = self.peek() {
                        if at == ")" {
                            break;
                        }
                        if at == "," {
                            self.consume();
                            continue;
                        }
                        args.push(self.consume().unwrap());
                    }
                    self.consume(); // )

                    for (i, arg_token) in args.iter().enumerate() {
                        let arg_op = self.parse_operand(arg_token);
                        let target_reg = (i + 1) as u8;
                        func.push(Instruction {
                            op: Opcode::Mov,
                            dest: Some(Operand::Reg(target_reg)),
                            src1: Some(arg_op),
                            src2: None,
                        });
                    }

                    func.push(Instruction {
                        op: Opcode::Call,
                        dest: None,
                        src1: Some(Operand::Label(func_name)),
                        src2: None,
                    });

                    let dest_reg = self.get_or_alloc_reg(&dest_name);
                    func.push(Instruction {
                        op: Opcode::Mov,
                        dest: Some(Operand::Reg(dest_reg)),
                        src1: Some(Operand::Reg(0)),
                        src2: None,
                    });
                } else if self
                    .peek()
                    .map(|s| "+-*/".contains(s) || s == "+" || s == "-")
                    .unwrap_or(false)
                {
                    // Binary Op
                    let op_str = self.consume().unwrap();
                    let token2 = self.consume().ok_or("Expected operand 2")?;

                    let src1 = self.parse_operand(&token1);
                    let src2 = self.parse_operand(&token2);
                    let dest_reg = self.get_or_alloc_reg(&dest_name);

                    func.push(Instruction {
                        op: Opcode::Mov,
                        dest: Some(Operand::Reg(dest_reg)),
                        src1: Some(src1),
                        src2: None,
                    });

                    let op = match op_str.as_str() {
                        "+" => Opcode::Add,
                        "-" => Opcode::Sub,
                        _ => return Err("Only + and - supported".to_string()),
                    };

                    func.push(Instruction {
                        op,
                        dest: Some(Operand::Reg(dest_reg)),
                        src1: Some(src2),
                        src2: None,
                    });
                } else {
                    // Simple imm/var: x = y
                    let src1 = self.parse_operand(&token1);
                    let dest_reg = self.get_or_alloc_reg(&dest_name);
                    func.push(Instruction {
                        op: Opcode::Mov,
                        dest: Some(Operand::Reg(dest_reg)),
                        src1: Some(src1),
                        src2: None,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::CodeGenerator;
    use crate::compiler::Compiler;
    use crate::jit_memory::DualMappedMemory;

    #[test]
    fn test_parse_and_run() {
        let script = "
            x = 10
            y = 32
            z = x + y
            return z
        ";
        let mut parser = Parser::new();
        let prog = parser.parse(script).expect("Parsing failed");
        let (code, main_offset) = Compiler::compile_program(&prog).expect("Compilation failed");

        // Emitting 'main' (last func usually? or we need entry point)
        // With compile_program, code contains ALL functions.
        // If implicit main, it is 'fn_main'.
        // We need entry point.
        // Just jumping to offset 0 works if 'main' is first?
        // compile_program iterates functions. Order matters.
        // Implicit main is added.

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code, 0);
        let func_ptr: extern "C" fn() -> i64 =
            unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
        assert_eq!(func_ptr(), 42);
    }

    #[test]
    fn test_loop_sum() {
        let script = "sum = 0
            i = 10
            label loop
            if i == 0 goto end
            sum = sum + i
            i = i - 1
            goto loop
            label end
            return sum";
        let mut parser = Parser::new();
        let prog = parser.parse(script).expect("Parsing failed");
        let code = Compiler::compile_program(&prog).expect("Compilation failed");
        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code.0, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 55);
    }

    #[test]
    fn test_function_call() {
        // fn add(a, b) { return a + b }
        // main: return add(10, 20)
        let script = "
            fn main() {
                x = add(10, 20)
                return x
            }
            fn add(a, b) {
                c = a + b
                return c
            }
        ";
        let mut parser = Parser::new();
        let prog = parser.parse(script).expect("Parsing failed");
        let code = Compiler::compile_program(&prog).expect("Compilation failed");
        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code.0, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 30);
    }
}
