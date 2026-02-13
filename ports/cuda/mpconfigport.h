// Configuration for CUDA port of MicroPython

#include <stdint.h>

// C++ compatibility for nvcc (which compiles as C++)
// The 'restrict' keyword is C99 but not valid C++
#ifdef __cplusplus
#define restrict
#endif

// Use minimal ROM level - disable most features
#define MICROPY_CONFIG_ROM_LEVEL        (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)

// CUDA-incompatible features - DISABLE
#ifndef MICROPY_NLR_SETJMP
#define MICROPY_NLR_SETJMP              (1)  // We provide custom implementation
#endif
#define MICROPY_ENABLE_GC               (0)  // No GC for POC. Our malloc implementation overr
#define MICROPY_STACK_CHECK             (0)  // No stack checking
#define MICROPY_OPT_COMPUTED_GOTO       (0)  // Use switch statement
#define MICROPY_NO_ALLOCA               (1)  // Don't use alloca
#define MICROPY_ENABLE_EXTERNAL_IMPORT  (0)  // No file imports
#define MICROPY_READER_POSIX            (0)  // No file reading
#define MICROPY_READER_VFS              (0)  // No VFS

// Compiler required for parsing Python code
#define MICROPY_ENABLE_COMPILER         (1)

// Disable REPL for POC
#define MICROPY_HELPER_REPL             (0)
#define MICROPY_REPL_EVENT_DRIVEN       (0)

// Minimal sys module
#define MICROPY_PY_SYS_MODULES          (0)
#define MICROPY_PY_SYS_EXIT             (0)
#define MICROPY_PY_SYS_PATH             (0)
#define MICROPY_PY_SYS_ARGV             (0)

// Disable all optional builtins
#define MICROPY_PY_BUILTINS_HELP        (0)
#define MICROPY_PY_BUILTINS_INPUT       (0)

// Memory settings - small allocations for POC
#define MICROPY_ALLOC_PATH_MAX          (64)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT  (16)

// Type definitions for the target
typedef long mp_off_t;

// Port state
#define MP_STATE_PORT MP_STATE_VM

// Board/MCU names
#define MICROPY_HW_BOARD_NAME           "cuda"
#define MICROPY_HW_MCU_NAME             "nvptx"

// We don't need frozen modules for POC
#define MICROPY_MODULE_FROZEN_MPY       (0)
#define MICROPY_MODULE_FROZEN_STR       (0)

// Use simple heap for memory allocation (since GC is disabled)
#define MICROPY_HEAP_SIZE               (16384)  // 16KB heap

// Port-specific function annotation
#ifdef __CUDA_ARCH__
#define MAYBE_CUDA __device__
#else
#define MAYBE_CUDA
#endif

// Device-side string function replacements for CUDA
#include "py/cuda_string.h"
