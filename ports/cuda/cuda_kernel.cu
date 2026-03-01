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
#include "py/mpstate.h"

#include "tests.h"
#define N_THREADS 2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Multi-thread MicroPython kernel
__global__ void micropython_kernel(int *results,
                                    mp_state_ctx_t *state_array,
                                    char *heap_base) {
    int tid = threadIdx.x;
    printf("[%d] micropython_kernel started\n", tid);

    // Each thread gets its own heap region
    char *my_heap = heap_base + tid * BUMP_ALLOC_HEAP_SIZE;

    // One call does everything: sets up state, allocator, and MicroPython
    mp_init(&state_array[tid], my_heap, BUMP_ALLOC_HEAP_SIZE);
    run_micropython_tests();
    mp_deinit();

    printf("[%d] micropython_kernel finished\n", tid);
    results[tid] = 42;
}
 
// Host function to launch the multi-thread test
extern "C" void run_cuda_test(void) {
    printf("CUDA MicroPython multi-thread test starting (%d threads)...\n", N_THREADS);

    // Set GPU stack size (MicroPython needs deep stacks)
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 16*1024));
     
    // Allocate per-thread result array
    int *d_results;
    gpuErrchk(cudaMalloc(&d_results, N_THREADS * sizeof(int)));

    // Allocate per-thread state contexts
    mp_state_ctx_t *d_states;
    gpuErrchk(cudaMalloc(&d_states, N_THREADS * sizeof(mp_state_ctx_t)));

    // Allocate per-thread heaps (contiguous block, each thread gets a slice)
    char *d_heap;
    gpuErrchk(cudaMalloc(&d_heap, N_THREADS * BUMP_ALLOC_HEAP_SIZE));

    // Launch kernel with N threads
    micropython_kernel<<<1, N_THREADS>>>(d_results, d_states, d_heap);
    gpuErrchk(cudaDeviceSynchronize());
     
    // Copy results back and print
    int h_results[N_THREADS];
    cudaMemcpy(h_results, d_results, N_THREADS * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N_THREADS; i++) {
        printf("Thread %d returned: %d\n", i, h_results[i]);
    }

    // Cleanup
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_heap);
}
