use crate::ir::{Function, Instruction, Opcode, Operand, Program};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Token {
    pub content: String,
    pub line: usize,
    pub col: usize,
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    symbol_table: HashMap<String, u8>, // Per-function symbol table
    next_reg: u8,
    label_counter: usize,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            pos: 0,
            symbol_table: HashMap::new(),
            next_reg: 1,
            label_counter: 0,
        }
    }

    fn tokenize(source: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = source.chars().collect();
        let mut i = 0;
        let mut line = 1;
        let mut col = 1;

        while i < chars.len() {
            let c = chars[i];

            if c == '#' {
                // Comment: skip until newline
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                // Newline consumption handled by loop or next iteration
                continue;
            }

            if c == '\n' {
                if !current.is_empty() {
                    tokens.push(Token {
                        content: current.clone(),
                        line,
                        col: col - current.len(),
                    });
                    current.clear();
                }
                line += 1;
                col = 1;
                i += 1;
                continue;
            }

            if c.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(Token {
                        content: current.clone(),
                        line,
                        col: col - current.len(),
                    });
                    current.clear();
                }
                i += 1;
                col += 1;
            } else if "(){},=+-[]:;<>!".contains(c) {
                if !current.is_empty() {
                    tokens.push(Token {
                        content: current.clone(),
                        line,
                        col: col - current.len(),
                    });
                    current.clear();
                }
                // Check for ==, !=, <=, >=
                if i + 1 < chars.len() {
                    let next = chars[i + 1];
                    if (c == '=' || c == '!' || c == '<' || c == '>') && next == '=' {
                        tokens.push(Token {
                            content: format!("{}{}", c, next),
                            line,
                            col,
                        });
                        i += 2;
                        col += 2;
                        continue;
                    }
                }
                tokens.push(Token {
                    content: c.to_string(),
                    line,
                    col,
                });
                i += 1;
                col += 1;
            } else {
                current.push(c);
                i += 1;
                col += 1;
            }
        }
        if !current.is_empty() {
            tokens.push(Token {
                content: current,
                line,
                col: col, // approx
            });
        }
        tokens
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
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
        if t.content == expected {
            Ok(())
        } else {
            Err(format!(
                "Expected '{}', found '{}' at line {}:{}",
                expected, t.content, t.line, t.col
            ))
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

    fn parse_operand(&mut self, token: &Token) -> Operand {
        if let Ok(num) = token.content.parse::<i32>() {
            Operand::Imm(num)
        } else {
            let reg = self.get_or_alloc_reg(&token.content);
            Operand::Reg(reg)
        }
    }

    fn generate_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        format!("{}_{}", prefix, self.label_counter)
    }

    pub fn parse(&mut self, source: &str) -> Result<Program, String> {
        self.tokens = Self::tokenize(source);
        self.pos = 0;
        let mut program = Program::new();

        while self.peek().is_some() {
            if self.peek().unwrap().content == "fn" {
                program.add_function(self.parse_function()?);
            } else {
                let t = self.peek().unwrap();
                return Err(format!(
                    "Unexpected token '{}' at line {}:{}. Top-level code is not allowed. Wrap in 'fn main() {{ ... }}'.",
                    t.content, t.line, t.col
                ));
            }
        }

        // Check for entry point
        let has_main = program.functions.iter().any(|f| f.name == "main");
        if !has_main {
            return Err("Missing entry point: fn main() not found".to_string());
        }

        Ok(program)
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        self.expect("fn")?;
        // Reset symbol table for new function
        self.symbol_table.clear();
        self.next_reg = 10; // Reserve 0..9 for Special/Phys Regs

        let name = self.consume().ok_or("Expected function name")?;
        self.expect("(")?;

        let mut args = Vec::new();
        while let Some(t) = self.peek() {
            if t.content == ")" {
                break;
            }
            if t.content == "," {
                self.consume();
                continue;
            }
            let arg_token = self.consume().unwrap();
            args.push(arg_token.content);
        }
        self.consume(); // )
        self.expect("{")?;

        let mut func = Function::new(&name.content, args.clone());

        // Emit Moves for Args
        for (i, arg_name) in args.iter().enumerate() {
            let user_reg = self.get_or_alloc_reg(arg_name);
            func.push(Instruction {
                op: Opcode::LoadArg(i),
                dest: Some(Operand::Reg(user_reg)),
                src1: None,
                src2: None,
            });
        }

        while let Some(t) = self.peek() {
            if t.content == "}" {
                self.consume();
                return Ok(func);
            }
            self.parse_statement(&mut func)?;
        }
        Err("Unexpected end of function".to_string())
    }

    fn parse_block(&mut self, func: &mut Function) -> Result<(), String> {
        self.expect("{")?;
        while let Some(t) = self.peek() {
            if t.content == "}" {
                self.consume();
                return Ok(());
            }
            self.parse_statement(func)?;
        }
        Err("Expected '}'".to_string())
    }

    // Helper to parse binary or simple assignment expressions
    // Currently specialized for simple cases required by loops
    // Returns the register where result is stored
    fn parse_expression(&mut self, func: &mut Function, dest_name: &str) -> Result<u8, String> {
         let token1 = self.consume().ok_or("Expected RHS")?;

         // Check Binary Op
         if let Some(next) = self.peek() {
              if "+-*/".contains(&next.content) || next.content == "+" || next.content == "-" {
                   let op_str = self.consume().unwrap();
                   let token2 = self.consume().ok_or("Expected operand 2")?;

                   let src1 = self.parse_operand(&token1);
                   let src2 = self.parse_operand(&token2);
                   let dest_reg = self.get_or_alloc_reg(dest_name);

                   func.push(Instruction {
                       op: Opcode::Mov,
                       dest: Some(Operand::Reg(dest_reg)),
                       src1: Some(src1),
                       src2: None,
                   });

                   let op = match op_str.content.as_str() {
                       "+" => Opcode::Add,
                       "-" => Opcode::Sub,
                       "*" => Opcode::Mul,
                       _ => return Err("Only +, -, and * supported".to_string()),
                   };

                   func.push(Instruction {
                       op,
                       dest: Some(Operand::Reg(dest_reg)),
                       src1: Some(src2),
                       src2: None,
                   });
                   return Ok(dest_reg);
              }
         }

         // Simple Assign
         let src1 = self.parse_operand(&token1);
         let dest_reg = self.get_or_alloc_reg(dest_name);
         func.push(Instruction {
             op: Opcode::Mov,
             dest: Some(Operand::Reg(dest_reg)),
             src1: Some(src1),
             src2: None,
         });
         Ok(dest_reg)
    }

    fn parse_statement(&mut self, func: &mut Function) -> Result<(), String> {
        let t = self.consume().ok_or("Unexpected EOF")?;

        match t.content.as_str() {
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
                    dest: Some(Operand::Label(name.content)),
                    src1: None,
                    src2: None,
                });
            }
            "goto" => {
                let name = self.consume().ok_or("Expected goto label")?;
                func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(name.content)),
                    src1: None,
                    src2: None,
                });
            }
            "while" => {
                // while cond { body }
                // Desugar:
                // label start
                // if !cond goto end (inverse logic hard, so we do: if cond goto body; goto end; label body...)
                // Simpler:
                // label start
                // parse_condition_jump_to_end?
                // Our IF is "if cond goto label".
                
                // Pattern:
                // label start
                // if cond goto body_label
                // goto end
                // label body_label
                // ... body ...
                // goto start
                // label end

                // Better Pattern (Standard):
                // label start
                // [evaluate cond -> tmp] -- Wait, our IF expects "x relative y"
                // if x == y goto body_start  <-- Logical inverse needed to jump to end?
                // We have Jump Not Zero (Jnz) if we had a bool.
                // But we support `if x < y goto`.
                // Let's invert the condition?
                // == -> !=
                // < -> >=
                // etc.
                
                let start_label = self.generate_label("while_start");
                let body_label = self.generate_label("while_body");
                let end_label = self.generate_label("while_end");

                // Label Start
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(start_label.clone())),
                    src1: None,
                    src2: None,
                });

                // Condition: "x < y"
                let lhs_token = self.consume().ok_or("Expected while condition lhs")?;
                let op_token = self.consume().ok_or("Expected while condition op")?;
                let rhs_token = self.consume().ok_or("Expected while condition rhs")?;

                let lhs = self.parse_operand(&lhs_token);
                let rhs = self.parse_operand(&rhs_token);

                func.push(Instruction {
                    op: Opcode::Cmp,
                    dest: None,
                    src1: Some(lhs),
                    src2: Some(rhs),
                });

                // Jump to Body if True
                let jump_op = match op_token.content.as_str() {
                    "==" => Opcode::Je,
                    "!=" => Opcode::Jne,
                    "<" => Opcode::Jl,
                    "<=" => Opcode::Jle,
                    ">" => Opcode::Jg,
                    ">=" => Opcode::Jge,
                    _ => return Err(format!("Unknown op {}", op_token.content)),
                };
                func.push(Instruction {
                    op: jump_op,
                    dest: Some(Operand::Label(body_label.clone())),
                    src1: None,
                    src2: None,
                });

                // False? Goto End
                func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(end_label.clone())),
                    src1: None,
                    src2: None,
                });

                // Body
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(body_label)),
                    src1: None,
                    src2: None,
                });

                self.parse_block(func)?;

                // Loop back
                func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(start_label)),
                    src1: None,
                    src2: None,
                });

                // End
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(end_label)),
                    src1: None,
                    src2: None,
                });
            }
            "for" => {
                // for (i=0; i<10; i=i+1) { ... }
                self.expect("(")?;

                // Init: i=0
                // Expect "var = val" or "var = expr"
                // parse_statement expects start of statement. 
                // But we are inside parens. 
                // Let's implement inline assignment parsing here.
                let init_var = self.consume().ok_or("Expected init var")?;
                self.expect("=")?;
                // Handle expression:
                let _ = self.parse_expression(func, &init_var.content)?;
                
                self.expect(";")?;
                
                let start_label = self.generate_label("for_start");
                let body_label = self.generate_label("for_body");
                let end_label = self.generate_label("for_end");
                let step_label = self.generate_label("for_step");

                // Label Start
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(start_label.clone())),
                    src1: None,
                    src2: None,
                });

                // Cond: i < 10
                let lhs_token = self.consume().ok_or("Expected cond lhs")?;
                let op_token = self.consume().ok_or("Expected cond op")?;
                let rhs_token = self.consume().ok_or("Expected cond rhs")?;
                
                let lhs = self.parse_operand(&lhs_token);
                let rhs = self.parse_operand(&rhs_token);

                func.push(Instruction {
                    op: Opcode::Cmp,
                    dest: None,
                    src1: Some(lhs),
                    src2: Some(rhs),
                });
                
                let jump_op = match op_token.content.as_str() {
                    "==" => Opcode::Je,
                    "!=" => Opcode::Jne,
                    "<" => Opcode::Jl,
                    "<=" => Opcode::Jle,
                    ">" => Opcode::Jg,
                    ">=" => Opcode::Jge,
                    _ => return Err(format!("Unknown op {} at line {}:{}", op_token.content, op_token.line, op_token.col)),
                };

                 // True -> Body
                func.push(Instruction {
                    op: jump_op,
                    dest: Some(Operand::Label(body_label.clone())),
                    src1: None,
                    src2: None,
                });
                
                // False -> End
                 func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(end_label.clone())),
                    src1: None,
                    src2: None,
                });

                self.expect(";")?;

                // Step: i = i + 1. 
                // We need to parse it but EMIT it AFTER body.
                // We can capture tokens and parse later? 
                // Or: emit to a temporary buffer?
                // Easier: Emit a JUMP to Body. Then Emit Step. Then JUMP to Start. 
                // Label Step is where Body jumps to.
                
                // Oops, standard: 
                // Start: Check
                // If True -> Body
                // If False -> End
                // Body: ... goto Step
                // Step: ... goto Start
                
                // So now we parse Step instructions immediately, but we are in the middle of emitting.
                // We can't easily "hold" instructions in this architecture without buffering.
                // Let's buffer the step tokens!
                
                let mut step_tokens = Vec::new();
                while let Some(t) = self.peek() {
                    if t.content == ")" {
                        break;
                    }
                    step_tokens.push(self.consume().unwrap());
                }
                self.expect(")")?;
                
                // Parse Body
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(body_label)),
                    src1: None,
                    src2: None,
                });
                
                self.parse_block(func)?;
                
                // Emit Step (replay tokens)
                // We need a sub-parser or just manually handle simple assignment
                // Assuming simple `var = expr`.
                if !step_tokens.is_empty() {
                    // Expect: var = rhs...
                    // Manual parsing of the buffer
                     let dest_name = &step_tokens[0].content;
                     if step_tokens[1].content != "=" {
                         return Err("Expected = in step".to_string());
                     }
                     // Everything after "=" is expression
                     // Hacky: Make a temporary parser?
                     // Or just implement simplified expression parser manually here.
                     
                     // Assuming `i = i + 1` (5 tokens: i, =, i, +, 1)
                     if step_tokens.len() == 3 {
                         // i = 1
                         let src = self.parse_operand(&step_tokens[2]);
                         let reg = self.get_or_alloc_reg(dest_name);
                          func.push(Instruction {
                            op: Opcode::Mov,
                            dest: Some(Operand::Reg(reg)),
                            src1: Some(src),
                            src2: None,
                        });
                     } else if step_tokens.len() == 5 {
                         let src1 = self.parse_operand(&step_tokens[2]);
                         let op_str = &step_tokens[3].content;
                         let src2 = self.parse_operand(&step_tokens[4]);
                         let reg = self.get_or_alloc_reg(dest_name);
                         
                         func.push(Instruction {
                            op: Opcode::Mov,
                            dest: Some(Operand::Reg(reg)),
                            src1: Some(src1),
                            src2: None,
                        });
                        let op = match op_str.as_str() {
                           "+" => Opcode::Add,
                           "-" => Opcode::Sub,
                           "*" => Opcode::Mul,
                           _ => return Err("Only +, -, * in loop step".to_string()),
                        };
                         func.push(Instruction {
                            op,
                            dest: Some(Operand::Reg(reg)),
                            src1: Some(src2),
                            src2: None,
                        });
                     } else {
                         return Err("Complex step not supported yet".to_string());
                     }
                }

                // Loop back
                func.push(Instruction {
                    op: Opcode::Jmp,
                    dest: Some(Operand::Label(start_label)),
                    src1: None,
                    src2: None,
                });

                // End
                func.push(Instruction {
                    op: Opcode::Label,
                    dest: Some(Operand::Label(end_label)),
                    src1: None,
                    src2: None,
                });
            }
            "free" => {
                self.expect("(")?;
                let ptr_token = self.consume().ok_or("Expected pointer")?;
                let ptr_op = self.parse_operand(&ptr_token);
                self.expect(")")?;
                func.push(Instruction {
                    op: Opcode::Free,
                    dest: None,
                    src1: Some(ptr_op),
                    src2: None,
                });
            }
            "if" => {
                let lhs_token = self.consume().ok_or("Expected if condition")?;
                let next = self.consume().ok_or("Expected if op or goto")?;

                if next.content == "goto" || next.content == "{" {
                    panic!("Simple 'if x goto' or block if' not supported fully. Use 'if x == y goto L' or 'if x == y {{ }}' ");
                } else {
                    let op_str = next.content;
                    let rhs_token = self.consume().ok_or("Expected rhs")?;
                    let action = self.consume().ok_or("Expected goto or {")?;
                    
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
                        _ => return Err(format!("Unknown op {} at line {}:{}", op_str, next.line, next.col)),
                    };
                    
                    if action.content == "goto" {
                         let label = self.consume().ok_or("Expected label")?;
                         func.push(Instruction {
                            op: jump_op,
                            dest: Some(Operand::Label(label.content)),
                            src1: None,
                            src2: None,
                        });
                    } else if action.content == "{" {
                        // if x == y { ... }
                        // Desugar: 
                        // if x == y goto block
                        // goto end
                        // label block
                        // ...
                        // label end
                        // Wait, easier: Inverse condition jump to end.
                        // But we already have the jump_op (e.g. Je).
                        // If Je, we want to run block.
                        // So: Je block_start; Jmp block_end; Label block_start; ... Label block_end
                        
                        let body_label = self.generate_label("if_body");
                        let end_label = self.generate_label("if_end");
                        
                        func.push(Instruction {
                            op: jump_op,
                            dest: Some(Operand::Label(body_label.clone())),
                            src1: None,
                            src2: None,
                        });
                         func.push(Instruction {
                            op: Opcode::Jmp,
                            dest: Some(Operand::Label(end_label.clone())),
                            src1: None,
                            src2: None,
                        });
                        
                        func.push(Instruction {
                            op: Opcode::Label,
                            dest: Some(Operand::Label(body_label.clone())),
                            src1: None,
                            src2: None,
                        });
                        
                        // Parse Block (already consumed {)
                         while let Some(t) = self.peek() {
                            if t.content == "}" {
                                self.consume();
                                break;
                            }
                            self.parse_statement(func)?;
                        }
                        
                         func.push(Instruction {
                            op: Opcode::Label,
                            dest: Some(Operand::Label(end_label.clone())),
                            src1: None,
                            src2: None,
                        });
                    } else {
                        return Err("Expected 'goto' or '{'".to_string());
                    }
                }
            }
            _ => {
                let dest_name = t.content;

                // Label: `name:`
                if let Some(next) = self.peek() {
                    if next.content == ":" {
                        self.consume(); // :
                        func.push(Instruction {
                            op: Opcode::Label,
                            dest: Some(Operand::Label(dest_name)),
                            src1: None,
                            src2: None,
                        });
                        return Ok(());
                    }
                }

                // Array Store: `dest[i] = val`
                if let Some(next) = self.peek() {
                    if next.content == "[" {
                        self.consume(); // [
                        let index_token = self.consume().ok_or("Expected index")?;
                        let index_op = self.parse_operand(&index_token);
                        self.expect("]")?;
                        self.expect("=")?;
                        let val_token = self.consume().ok_or("Expected value")?;
                        let val_op = self.parse_operand(&val_token);
                        let base_reg = self.get_or_alloc_reg(&dest_name);

                        func.push(Instruction {
                            op: Opcode::Store,
                            dest: Some(Operand::Reg(base_reg)),
                            src1: Some(index_op),
                            src2: Some(val_op),
                        });
                        return Ok(());
                    }
                }

                let eq = self.consume().ok_or("Expected =")?;
                if eq.content != "=" {
                    return Err(format!("Expected =, found {} at line {}:{}", eq.content, eq.line, eq.col));
                }

                let token1 = self.consume().ok_or("Expected RHS")?;

                // Array Load: `y = x[i]`
                if let Some(next) = self.peek() {
                    if next.content == "[" {
                        self.consume(); // [
                        let index_token = self.consume().ok_or("Expected index")?;
                        let index_op = self.parse_operand(&index_token);
                        self.expect("]")?;

                        let base_reg = self.get_or_alloc_reg(&token1.content);
                        let dest_reg = self.get_or_alloc_reg(&dest_name);

                        func.push(Instruction {
                            op: Opcode::Load,
                            dest: Some(Operand::Reg(dest_reg)),
                            src1: Some(Operand::Reg(base_reg)),
                            src2: Some(index_op),
                        });
                        return Ok(());
                    }
                }

                // Function Call: `y = func(...)`
                if let Some(next) = self.peek() {
                    if next.content == "(" {
                        self.consume(); // (
                        
                        if token1.content == "alloc" {
                            let size_token = self.consume().ok_or("Expected size")?;
                            let size_op = self.parse_operand(&size_token);
                            self.expect(")")?;
                            let dest_reg = self.get_or_alloc_reg(&dest_name);
                            func.push(Instruction {
                                op: Opcode::Alloc,
                                dest: Some(Operand::Reg(dest_reg)),
                                src1: Some(size_op),
                                src2: None,
                            });
                            return Ok(());
                        }

                        let mut args = Vec::new();
                        while let Some(t) = self.peek() {
                            if t.content == ")" {
                                break;
                            }
                            if t.content == "," {
                                self.consume();
                                continue;
                            }
                            let arg_tok = self.consume().unwrap();
                            args.push(self.parse_operand(&arg_tok));
                        }
                        self.expect(")")?;

                        for (i, arg) in args.iter().enumerate() {
                            let arg_phys_vreg = (i + 1) as u8;
                            func.push(Instruction {
                                op: Opcode::SetArg(i),
                                dest: Some(Operand::Reg(arg_phys_vreg)),
                                src1: Some(arg.clone()),
                                src2: None,
                            });
                        }

                        let dest_reg = self.get_or_alloc_reg(&dest_name);
                        func.push(Instruction {
                            op: Opcode::Call,
                            dest: Some(Operand::Reg(dest_reg)),
                            src1: Some(Operand::Label(token1.content)),
                            src2: None,
                        });
                        return Ok(());
                    }
                }

                // Binary Op: `y = a + b`
                if let Some(next) = self.peek() {
                    if "+-*/".contains(&next.content) || next.content == "+" || next.content == "-" {
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
     
                         let op = match op_str.content.as_str() {
                             "+" => Opcode::Add,
                             "-" => Opcode::Sub,
                             "*" => Opcode::Mul,
                             _ => return Err("Only +, -, and * supported".to_string()),
                         };
     
                         func.push(Instruction {
                             op,
                             dest: Some(Operand::Reg(dest_reg)),
                             src1: Some(src2),
                             src2: None,
                         });
                         return Ok(());
                    }
                }

                // Simple Assign: `y = x`
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
        Ok(())
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::assembler::CodeGenerator;
    #[allow(unused_imports)]
    use crate::compiler::Compiler;
    #[allow(unused_imports)]
    use crate::jit_memory::DualMappedMemory;

    #[test]
    fn test_parse_and_run() {
        let script = "
            fn main() {
                x = 10
                y = 32
                z = x + y
                return z
            }
        ";
        let mut parser = Parser::new();
        let prog = parser.parse(script).expect("Parsing failed");
        let (code, main_offset) = Compiler::compile_program(&prog, 0).expect("Compilation failed");

        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code, 0);
        let func_ptr: extern "C" fn() -> i64 =
            unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
        assert_eq!(func_ptr(), 42);
    }

    #[test]
    fn test_loop_sum() {
        // Updated to use while loop sugar
        let script = "
            fn main() {
                sum = 0
                i = 10
                while i > 0 {
                    sum = sum + i
                    i = i - 1
                }
                return sum
            }
        ";
        let mut parser = Parser::new();
        let prog = parser.parse(script).expect("Parsing failed");
        let code = Compiler::compile_program(&prog, 0).expect("Compilation failed");
        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code.0, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 55);
    }

    #[test]
    fn test_function_call() {
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
        let code = Compiler::compile_program(&prog, 0).expect("Compilation failed");
        let memory = DualMappedMemory::new(4096).unwrap();
        CodeGenerator::emit_to_memory(&memory, &code.0, 0);
        let func_ptr: extern "C" fn() -> i64 = unsafe { std::mem::transmute(memory.rx_ptr) };
        assert_eq!(func_ptr(), 30);
    }
}
