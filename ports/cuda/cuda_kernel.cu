// CUDA kernel wrapper for MicroPython
// This file is compiled with nvcc and wraps the C MicroPython library

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"

#include "tests.h"
#include "py/bumpalloc.h"

#define BUMP_ALLOC_HEAP_SIZE (100 * 1024)

// Simple test kernel
__global__ void micropython_kernel(int *result, char *heap_ptr) {
    printf("GPU: micropython_kernel started\n");

    // Initialize MicroPython
    bump_alloc_init(heap_ptr, BUMP_ALLOC_HEAP_SIZE);
    mp_init();
    
    // Run tests
    run_micropython_tests();
    
    mp_deinit();

    printf("GPU: micropython_kernel finished\n");
    // Signal success
    *result = 42;
}
 
// Host function to test the setup
extern "C" void run_cuda_test(void) {
    printf("CUDA MicroPython hybrid test starting...\n");
     
    // Allocate device memory
    int *d_result;
    int h_result = 0;
    cudaMalloc(&d_result, sizeof(int));

    // Allocate heap
    char *d_heap;
    cudaError_t err = cudaMalloc(&d_heap, BUMP_ALLOC_HEAP_SIZE);
    if (err != cudaSuccess) {
        printf("Failed to allocate device heap: %s\n", cudaGetErrorString(err));
        return;
    }
     
    // Launch a simple kernel
    micropython_kernel<<<1, 1>>>(d_result, d_heap);
    cudaDeviceSynchronize();
     
    // Copy result back
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_heap);
     
    printf("Kernel returned: %d\n", h_result);
}
