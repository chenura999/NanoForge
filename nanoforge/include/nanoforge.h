/**
 * NanoForge FFI - C-compatible API Header
 * 
 * A Self-Optimizing Assembly Engine with AI-Powered Variant Selection
 * 
 * Usage:
 *   1. Link with libnanoforge.so or libnanoforge.a
 *   2. Call nanoforge_init() to detect CPU features
 *   3. Use nanoforge_compile() to compile scripts
 *   4. Use nanoforge_optimizer_* for AI-powered variant selection
 */

#ifndef NANOFORGE_H
#define NANOFORGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct NanoFunction NanoFunction;
typedef struct NanoOptimizer NanoOptimizer;

/* Result codes */
typedef enum {
    NANO_OK = 0,
    NANO_ERROR_PARSE_FAILED = 1,
    NANO_ERROR_COMPILE_FAILED = 2,
    NANO_ERROR_NULL_POINTER = 3,
    NANO_ERROR_INVALID_UTF8 = 4,
    NANO_ERROR_IO_FAILED = 5,
} NanoResult;

/* ============================================================================
 * Core Functions
 * ============================================================================ */

/**
 * Initialize NanoForge and detect CPU features.
 * @return String describing CPU features (caller must free with nanoforge_free_string)
 */
char* nanoforge_init(void);

/**
 * Free a string returned by NanoForge.
 */
void nanoforge_free_string(char* s);

/**
 * Get NanoForge version.
 * @return Static string, do not free
 */
const char* nanoforge_version(void);

/* ============================================================================
 * Compilation Functions
 * ============================================================================ */

/**
 * Compile a NanoForge script.
 * @param source NanoForge source code (null-terminated)
 * @return Function handle (caller must free with nanoforge_free_function)
 */
NanoFunction* nanoforge_compile(const char* source);

/**
 * Execute a compiled function.
 * @param func Function handle
 * @param input Input value
 * @return Result of function execution
 */
uint64_t nanoforge_execute(const NanoFunction* func, uint64_t input);

/**
 * Free a compiled function.
 */
void nanoforge_free_function(NanoFunction* func);

/* ============================================================================
 * AI Optimizer Functions
 * ============================================================================ */

/**
 * Create a new AI optimizer (Contextual Bandit).
 */
NanoOptimizer* nanoforge_optimizer_new(void);

/**
 * Select variant using AI optimizer.
 * @param opt Optimizer handle
 * @param input_size Size of input data
 * @return Index of selected variant (-1 on error)
 */
int32_t nanoforge_optimizer_select(NanoOptimizer* opt, uint64_t input_size);

/**
 * Update AI optimizer with performance feedback.
 * @param opt Optimizer handle
 * @param input_size Size of input data
 * @param variant_idx Index of variant that was tested
 * @param cycles Cycles per operation achieved
 * @param best_cycles Best known cycles per operation
 */
void nanoforge_optimizer_update(
    NanoOptimizer* opt,
    uint64_t input_size,
    int32_t variant_idx,
    uint64_t cycles,
    uint64_t best_cycles
);

/**
 * Save AI optimizer state to file.
 * @param opt Optimizer handle
 * @param path File path (null-terminated)
 * @return Result code
 */
NanoResult nanoforge_optimizer_save(const NanoOptimizer* opt, const char* path);

/**
 * Load AI optimizer from file (or create new if not exists).
 * @param path File path (null-terminated)
 * @return Optimizer handle (caller must free with nanoforge_optimizer_free)
 */
NanoOptimizer* nanoforge_optimizer_load(const char* path);

/**
 * Free AI optimizer.
 */
void nanoforge_optimizer_free(NanoOptimizer* opt);

#ifdef __cplusplus
}
#endif

#endif /* NANOFORGE_H */
