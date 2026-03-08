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
#include "py/objarray.h"
#include "py/obj.h"

#include "ports/cuda/tests.h"

#ifdef __CUDACC__
extern "C" void run_cuda_test(void);
#endif

// Main function - can be called from CUDA kernel or host
int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("CUDA MicroPython POC starting...\n");

    #ifdef __CUDACC__
    run_cuda_test();
    #else
    // Host build: allocate state and heap, then call mp_init
    static mp_state_ctx_t host_state_ctx;
    char *heap_ptr = (char *)malloc(BUMP_ALLOC_HEAP_SIZE);
    if (!heap_ptr) {
        printf("FATAL: Failed to allocate heap\n");
        return 1;
    }

    mp_init(&host_state_ctx, heap_ptr, BUMP_ALLOC_HEAP_SIZE);
    run_micropython_tests();
    mp_deinit();

    free(heap_ptr);
    #endif

    printf("CUDA MicroPython POC finished.\n");
    return 0;
}

// Required stubs for MicroPython

// Lexer from file - not supported
MAYBE_CUDA mp_lexer_t *mp_lexer_new_from_file(qstr filename) {
    mp_raise_OSError(MP_ENOENT);
}

// Import stat - nothing exists
MAYBE_CUDA mp_import_stat_t mp_import_stat(const char *path) {
    return MP_IMPORT_STAT_NO_EXIST;
}

// NLR jump fail - called when an exception has no handler
// This is required by nlrsetjmp.c
MAYBE_CUDA void nlr_jump_fail(void *val) {
    printf("FATAL: Uncaught exception:\n");
    mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(val));
    #ifdef __CUDACC__
    asm("trap;");
    #endif
    while (1) { }
}

// Fatal error handler
MAYBE_CUDA void __fatal_error(const char *msg) {
    printf("FATAL ERROR: %s\n", msg);
    while (1) { }
}

// Bind an externally-owned byte array into the Python __main__ namespace.
// Creates a writable memoryview over [start, start+len) and stores it as
// a global named `name`.  The caller owns the memory; MicroPython never
// frees or resizes it.  Must be called after mp_init().
MAYBE_CUDA void mp_bind_array(const char *name, void *start, int len) {
    // Build a writable memoryview (typecode 'B' = uint8) over the external buffer.
    mp_obj_t mv = mp_obj_new_memoryview(
        'B' | MP_OBJ_ARRAY_TYPECODE_FLAG_RW,
        (size_t)len,
        start
    );

    // Store it in __main__'s globals dict under the given name.
    mp_obj_t key = mp_obj_new_str(name, strlen(name));
    mp_obj_dict_store(mp_module___main__.globals, key, mv);
}
