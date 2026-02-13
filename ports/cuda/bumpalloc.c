/*
 * Bump allocator globals and host-side initialization for CUDA.
 *
 * This file defines the device-side globals used by the bump allocator
 * and provides host-side initialization functions.
 */

#include <stdio.h>
#include <string.h>
#include "py/bumpalloc.h"

#ifdef __CUDACC__

// Device-side globals
__device__ char *bump_heap_base = 0;
__device__ size_t bump_heap_offset = 0;
__device__ size_t bump_heap_size = 0;

// Kernel to initialize device-side globals from a pointer allocated by cudaMalloc
__global__ void bump_alloc_init_kernel(char *heap_ptr, size_t heap_size) {
    bump_heap_base = heap_ptr;
    bump_heap_offset = 0;
    bump_heap_size = heap_size;
}

// Host-side function to initialize the bump allocator
// Returns the device heap pointer (caller must cudaFree it when done)
char *bump_alloc_init(size_t heap_size) {
    char *d_heap = 0;
    cudaError_t err = cudaMalloc((void **)&d_heap, heap_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA bump_alloc_init: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    // Clear the heap to zero
    cudaMemset(d_heap, 0, heap_size);

    // Initialize device globals
    bump_alloc_init_kernel<<<1, 1>>>(d_heap, heap_size);
    cudaDeviceSynchronize();

    return d_heap;
}

// Host-side function to clean up the bump allocator
void bump_alloc_deinit(char *d_heap) {
    if (d_heap) {
        cudaFree(d_heap);
    }
}

#else

// Host build: provide stub implementations
// On host, we just use a static buffer and bump through it
static char bump_host_heap[BUMP_ALLOC_HEAP_SIZE];
static size_t bump_host_offset = 0;

char *bump_alloc_init(size_t heap_size) {
    (void)heap_size;
    bump_host_offset = 0;
    memset(bump_host_heap, 0, sizeof(bump_host_heap));
    return bump_host_heap;
}

void bump_alloc_deinit(char *d_heap) {
    (void)d_heap;
}

#endif
