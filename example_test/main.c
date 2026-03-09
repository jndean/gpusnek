// Main entry point for CUDA MicroPython port
// This is a minimal implementation for POC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests.h"

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
    char *memory_ptr = (char *)malloc(PYSTACK_SIZE + HEAP_SIZE);
    if (!memory_ptr) {
        printf("FATAL: Failed to allocate memory\n");
        return 1;
    }

    mp_init(&host_state_ctx, memory_ptr, PYSTACK_SIZE, memory_ptr + PYSTACK_SIZE, HEAP_SIZE);

    #if MICROPY_ENABLE_GC
    // Set the C stack top for conservative GC root scanning. This must be
    // done manually here because mp_init clears the state context.
    int stack_dummy;
    MP_STATE_THREAD(stack_top) = (char *)&stack_dummy;
    #endif

    run_micropython_tests();
    mp_deinit();

    free(memory_ptr);
    #endif

    printf("CUDA MicroPython POC finished.\n");
    return 0;
}

