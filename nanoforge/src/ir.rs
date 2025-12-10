#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// A virtual register (0..N).
    /// For the trivial allocator, these map directly to hardware registers.
    Reg(u8),
    /// An immediate integer constant.
    Imm(i32),
    /// A label for jump targets.
    Label(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    /// Mov dest, src
    Mov,
    /// Add dest, src (dest += src)
    Add,
    /// Sub dest, src (dest -= src)
    Sub,
    /// Return the value in the first operand (or Accumulator/Reg(0))
    Ret,
    /// Define a label
    Label,
    /// Unconditional Jump
    Jmp,
    /// Jump if Not Zero (Legacy, kept for sugar or simple checks)
    Jnz,
    /// Compare two operands (sets flags)
    Cmp,
    /// Jump Equal
    Je,
    /// Jump Not Equal
    Jne,
    /// Jump Less
    Jl,
    /// Jump Less or Equal
    Jle,
    /// Jump Greater
    Jg,
    /// Jump Greater or Equal
    Jge,
    /// Call a function
    Call,
    /// Load Argument from Stack (index 0-based)
    LoadArg(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub op: Opcode,
    pub dest: Option<Operand>,
    pub src1: Option<Operand>,
    pub src2: Option<Operand>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub args: Vec<String>,
    pub instructions: Vec<Instruction>,
}

impl Function {
    pub fn new(name: &str, args: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            args,
            instructions: Vec::new(),
        }
    }

    pub fn push(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }
}

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: Function) {
        self.functions.push(func);
    }
}
