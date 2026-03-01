// Main entry point for CUDA MicroPython port
// This is a minimal implementation for POC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"

#include "ports/cuda/tests.h"
#include "py/bumpalloc.h"

#define BUMP_ALLOC_HEAP_SIZE (100 * 1024)

#ifdef __CUDACC__
extern "C" void run_cuda_test(void);
#endif

// Main function - can be called from CUDA kernel or host
int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("CUDA MicroPython POC starting...\n");

    // Initialize bump allocator (must happen before mp_init)
    #ifdef __CUDACC__
    // In CUDA build, main() constructs the heap and kernel initializes it.
    // But here main() calls run_cuda_test(), which handles allocation.
    // See run_cuda_test in cuda_kernel.cu
    #else
    // In Host simulation build: allocate state context and heap
    static mp_state_ctx_t host_state_ctx;
    memset(&host_state_ctx, 0, sizeof(host_state_ctx));
    mp_state_ctx_array = &host_state_ctx;

    static bump_alloc_state_t host_bump_state;
    bump_alloc_states = &host_bump_state;

    static mp_obj_module_t host_module_main;
    memset(&host_module_main, 0, sizeof(host_module_main));
    mp_module___main___array = &host_module_main;

    char *heap_ptr = (char *)malloc(BUMP_ALLOC_HEAP_SIZE);
    if (!heap_ptr) {
        printf("FATAL: Failed to allocate heap\n");
        return 1;
    }
    bump_alloc_init(heap_ptr, BUMP_ALLOC_HEAP_SIZE);
    #endif

    
    #ifdef __CUDACC__
    run_cuda_test();
    #else
    mp_init();
    run_micropython_tests();
    mp_deinit();
    #endif

    
    #ifndef __CUDACC__
    free(heap_ptr);
    #endif

    printf("CUDA MicroPython POC finished.\n");
    return 0;
}

// Required stubs for MicroPython

// Lexer from file - not supported
mp_lexer_t *mp_lexer_new_from_file(qstr filename) {
    mp_raise_OSError(MP_ENOENT);
}

// Import stat - nothing exists
mp_import_stat_t mp_import_stat(const char *path) {
    return MP_IMPORT_STAT_NO_EXIST;
}

// NLR jump fail - called when an exception has no handler
// This is required by nlrsetjmp.c
void nlr_jump_fail(void *val) {
    printf("FATAL: Uncaught NLR jump (exception with no handler)\n");
    while (1) { }
}

// Fatal error handler
void __fatal_error(const char *msg) {
    printf("FATAL ERROR: %s\n", msg);
    while (1) { }
}

#ifndef NDEBUG
void __assert_func(const char *file, int line, const char *func, const char *expr) {
    printf("Assertion '%s' failed, at file %s:%d\n", expr, file, line);
    __fatal_error("Assertion failed");
}
#endif
