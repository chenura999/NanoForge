#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operand {
    Reg(u8),       // Virtual Integer Register
    Ymm(u8),       // Virtual Vector Register (AVX2)
    Imm(i32),      // Immediate value
    Label(String), // Label name
}

#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    /// Mov dest, src
    Mov,
    /// Add dest, src (dest += src)
    Add,
    /// Mul dest, src (dest *= src)
    Mul,
    /// Sub dest, src (dest -= src)
    Sub,
    /// Return the value in the first operand (or Accumulator/Reg(0))
    Ret,
    /// Define a label
    Label,
    /// Unconditional Jump
    Jmp,
    /// Alloc(dest, size) -> dest = malloc(size)
    Alloc,
    /// Free(ptr) -> free(ptr)
    Free,
    /// Load(dest, base, index) -> dest = MEM[base + index * 8]
    Load,
    /// Store(base, index, src) -> MEM[base + index * 8] = src
    Store,
    SetArg(usize), // Set Argument i for Call
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
    /// VLoad(ymm_dest, base, index) -> ymm_dest = MEM[base + index * 8] (Vector Load)
    VLoad,
    /// VStore(base, index, ymm_src) -> MEM[base + index * 8] = ymm_src (Vector Store)
    VStore,
    /// VAdd(ymm_dest, ymm_src1, ymm_src2) -> ymm_dest = ymm_src1 + ymm_src2 (Packed Add)
    VAdd,
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

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}
