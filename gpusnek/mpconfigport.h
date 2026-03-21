// Configuration for CUDA port of MicroPython

#include <stdint.h>

// C++ compatibility for nvcc (which compiles as C++)
// The 'restrict' keyword is C99 but not valid C++
#ifdef __cplusplus
#define restrict
#endif

// Use minimal ROM level - disable most features
#define MICROPY_CONFIG_ROM_LEVEL        (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)

#define MICROPY_NLR_SETJMP              (1)  // We provide custom implementation
#define MICROPY_ENABLE_GC               (1)  // Enable garbage collector
#define MICROPY_ENABLE_PYSTACK          (1)  // Enable explicit python stack to trace GC roots natively
#define MICROPY_STACK_CHECK             (0)  // No stack checking
#define MICROPY_OPT_COMPUTED_GOTO       (0)  // Use switch statement
#define MICROPY_NO_ALLOCA               (1)  // Don't use alloca
#define MICROPY_ENABLE_EXTERNAL_IMPORT  (0)  // No file imports
#define MICROPY_READER_POSIX            (0)  // No file reading
#define MICROPY_READER_VFS              (0)  // Enable VFS reader
#define MICROPY_VFS                     (1)  // Enable VFS subsystem
#define MICROPY_VFS_LFS2                (1)  // Enable LittleFS v2
#define MICROPY_ENABLE_FINALISER        (1)  // Required by VFS LFS
#define MICROPY_PY_OS                   (1)  // Enable os module
#define MICROPY_PY_VFS                  (1)  // Enable VFS module
#define MICROPY_PY_IO                   (1)  // Enable open() builtin (via VFS)
// Compiler required for parsing Python code
#define MICROPY_ENABLE_COMPILER         (1)

// Enable event-driven REPL
#define MICROPY_HELPER_REPL             (1)
#define MICROPY_REPL_EVENT_DRIVEN       (1)

// Minimal sys module
#define MICROPY_PY_SYS_MODULES          (0)
#define MICROPY_PY_SYS_EXIT             (0)
#define MICROPY_PY_SYS_PATH             (0)
#define MICROPY_PY_SYS_ARGV             (0)

// Disable all optional builtins
#define MICROPY_PY_BUILTINS_HELP        (0)
#define MICROPY_PY_BUILTINS_INPUT       (0)
// Enable memoryview + array — required for gpusnek_bind_memory to wrap external buffers.
// memoryview depends on array_new() which is gated on MICROPY_PY_ARRAY.
#define MICROPY_PY_BUILTINS_MEMORYVIEW  (1)
#define MICROPY_PY_ARRAY                (1)
#define MICROPY_PY_BUILTINS_BYTEARRAY   (1)
#define MICROPY_PY_BUILTINS_SLICE       (1)
#define MICROPY_PY_ARRAY_SLICE_ASSIGN   (1)
#define MICROPY_FLOAT_IMPL              (MICROPY_FLOAT_IMPL_FLOAT)

// Memory settings - small allocations for POC
#define MICROPY_ALLOC_PATH_MAX          (64)
#define MICROPY_ALLOC_PARSE_CHUNK_INIT  (16)

#define MICROPY_USE_INTERNAL_PRINTF (0)

// Type definitions for the target
typedef long mp_off_t;

// Port state
#define MP_STATE_PORT MP_STATE_VM

// Board/MCU names
#define MICROPY_HW_BOARD_NAME           "gpu"
#define MICROPY_HW_MCU_NAME             "1,048,576 snek"

// We don't need frozen modules for POC
#define MICROPY_MODULE_FROZEN_MPY       (0)
#define MICROPY_MODULE_FROZEN_STR       (0)

// Use simple heap for memory allocation (since GC is disabled)
#define MICROPY_HEAP_SIZE               (16384)  // 16KB heap

// Port-specific function annotation
#ifdef __CUDACC__
#define MAYBE_CUDA __device__
#else
#define MAYBE_CUDA
#endif

// Device-side string function replacements for CUDA
#include "py/cuda_string.h"
