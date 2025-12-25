use nanoforge::compiler::Compiler;
use nanoforge::assembler::CodeGenerator;
use nanoforge::jit_memory::DualMappedMemory;
use nanoforge::parser::Parser as NanoParser;
use std::fs;
use std::path::Path;

fn run_test_file(path: &Path) -> Result<(), String> {
    println!("Running test: {:?}", path);
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    
    let mut parser = NanoParser::new();
    let prog = parser.parse(&content).map_err(|e| format!("Parse Error: {}", e))?;
    
    // Compile (Level 2 = Scalar)
    let (code, main_offset) = Compiler::compile_program(&prog, 2)
        .map_err(|e| format!("Compile Error: {}", e))?;
        
    let memory = DualMappedMemory::new(code.len() + 4096)
        .map_err(|e| format!("Memory Error: {}", e))?;
        
    CodeGenerator::emit_to_memory(&memory, &code, 0);
    
    let func_ptr: extern "C" fn() -> i64 = 
        unsafe { std::mem::transmute(memory.rx_ptr.add(main_offset)) };
        
    let result = func_ptr();
    
    if result == 0 {
        Ok(())
    } else {
        Err(format!("Test failed with exit code: {}", result))
    }
}

#[test]
fn run_all_programs() {
    let test_dir = Path::new("tests/programs");
    if !test_dir.exists() {
        // Fallback for running from inside 'nanoforge' dir or root
        if Path::new("nanoforge/tests/programs").exists() {
            // This handles if we are in the parent directory
            panic!("Please run cargo test from the nanoforge crate root directory"); 
        }
        panic!("tests/programs directory not found at {:?}", std::env::current_dir());
    }

    let mut failures = Vec::new();

    for entry in fs::read_dir(test_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("nf") {
            if let Err(e) = run_test_file(&path) {
                failures.push((path, e));
            }
        }
    }

    if !failures.is_empty() {
        for (path, err) in &failures {
            eprintln!("FAIL: {:?} -> {}", path, err);
        }
        panic!("{} tests failed.", failures.len());
    }
}
