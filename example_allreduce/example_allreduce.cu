#include <stdio.h>
#include <stdlib.h>

#include "gpusnek/gpusnek.h"

#define HEAP_SIZE (5 * 1024)
#define PYSTACK_SIZE (1 * 1024)
#define PER_THREAD_MEMORY (PYSTACK_SIZE + HEAP_SIZE)

#define N_THREADS 128
#define N_ELEMENTS 1024



__global__ void allreduce_kernel(int *d_data, mp_state_ctx_t *states, char *memory) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    char *thread_mem = memory + tid * PER_THREAD_MEMORY;
    gpusnek_init(states, thread_mem, PYSTACK_SIZE, HEAP_SIZE);


    gpusnek_bind_memory("shared_arr", d_data, N_ELEMENTS, 'i');
    gpusnek_new_int("tid", tid);
    gpusnek_do_str(
        "for step in range(10):\n"
        "    stride = 1 << step\n"
        "    work_size = 512 >> step\n"
        "    for w in range(tid, work_size, 128):\n"
        "        idx = w * (stride * 2)\n"
        "        shared_arr[idx] = shared_arr[idx] + shared_arr[idx + stride]\n"
        "    syncthreads\n"
    );
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}


int main(void) {
    printf(
        "Starting Parallel All-Reduce Example (%d elements / %d threads)\n", 
        N_ELEMENTS, N_THREADS
    );
    
    int *d_data;
    int *h_data = (int *)malloc(N_ELEMENTS * sizeof(int));
    for (int i = 0; i < N_ELEMENTS; i++) {
        h_data[i] = i;
    }
    int expected_sum = (N_ELEMENTS * (N_ELEMENTS - 1)) / 2;
    
    mp_state_ctx_t *d_states;
    char *d_memory;
    gpuErrchk(cudaMalloc(&d_states, N_THREADS * sizeof(mp_state_ctx_t)));
    gpuErrchk(cudaMalloc(&d_memory, N_THREADS * (PYSTACK_SIZE + HEAP_SIZE)));
    gpuErrchk(cudaMalloc(&d_data, N_ELEMENTS * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_data, h_data, N_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 5*1024));



    // Launch single block of 128 threads that will perform the parallel sum
    allreduce_kernel<<<1, N_THREADS>>>(d_data, d_states, d_memory);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result: %d\n", h_data[0]);

    if (h_data[0] == expected_sum) printf("SUCCESS!\n");
    else                           printf("FAILED!\n");


    cudaFree(d_states);
    cudaFree(d_memory);
    cudaFree(d_data);
    free(h_data);
    return 0;
}
