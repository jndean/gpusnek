#include <stdio.h>
#include <stdlib.h>

#include "gpusnek/gpusnek.h"
#include "utils_for_examples.h"

#define HEAP_SIZE (5 * 1024)
#define PYSTACK_SIZE (1 * 1024)
#define PER_THREAD_MEMORY (PYSTACK_SIZE + HEAP_SIZE)

#define N_ELEMENTS 256

#ifdef __CUDACC__

__global__ void mean_kernel(float *d_data, mp_state_ctx_t *states, char *memory) {
    int tid = threadIdx.x;
    __shared__ float cache[N_ELEMENTS];

    char *thread_mem = memory + tid * PER_THREAD_MEMORY;
    gpusnek_init(states, thread_mem, PYSTACK_SIZE, HEAP_SIZE);

    gpusnek_bind_memory("data", d_data, N_ELEMENTS, 'f');
    gpusnek_bind_memory("cache", cache, N_ELEMENTS, 'f');
    gpusnek_new_int("tid", tid);
    gpusnek_new_int("N", N_ELEMENTS);
    gpusnek_do_str(R"(
from math import log

cache[tid] = data[tid]

steps = int(log(N, 2))
for i in range(steps):
    offset = N >> (i + 1)
    if tid < offset:
        cache[tid] += cache[tid + offset]
    syncthreads

if tid == 0:
    data[0] = cache[0] / len(data)
)");
}


int main(void) {
    printf(
        "Starting Parallel All-Reduce Example (%d elements / %d threads)\n",
        N_ELEMENTS, N_ELEMENTS
    );

    float *d_data;
    float *h_data = (float *)malloc(N_ELEMENTS * sizeof(int));
    for (int i = 0; i < N_ELEMENTS; i++) {
        h_data[i] = i;
    }
    float expected_mean = (N_ELEMENTS - 1) / 2.;

    mp_state_ctx_t *d_states;
    char *d_memory;
    catchError(cudaMalloc(&d_states, N_ELEMENTS * sizeof(mp_state_ctx_t)));
    catchError(cudaMalloc(&d_memory, N_ELEMENTS * (PYSTACK_SIZE + HEAP_SIZE)));
    catchError(cudaMalloc(&d_data, N_ELEMENTS * sizeof(float)));
    catchError(cudaMemcpy(d_data, h_data, N_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    catchError(cudaDeviceSetLimit(cudaLimitStackSize, 5*1024));


    mean_kernel<<<1, N_ELEMENTS>>>(d_data, d_states, d_memory);
    catchError(cudaGetLastError());
    catchError(cudaDeviceSynchronize());

    catchError(cudaMemcpy(h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Result: %f\n", h_data[0]);

    if (h_data[0] == expected_mean) printf("SUCCESS!\n");
    else                           printf("FAILED! Expected %f\n", expected_mean);

    cudaFree(d_states);
    cudaFree(d_memory);
    cudaFree(d_data);
    free(h_data);
    return 0;
}

#else // !__CUDACC__

int main(void) {
    printf("example_allreduce: CUDA-only example, not supported on host.\n");
    return 0;
}

#endif // __CUDACC__





// __global__ 
// void average_kernel(float *data) {
//     int threadId = threadIdx.x;
//     __shared__ float cache[NUM_THREADS];
    
//     gpusnek_bind_memory("cache", cache, NUM_THREADS, 'f');
//     gpusnek_bind_memory("data", data, NUM_THREADS, 'f');
//     gpusnek_new_int("tid", threadId);
//     gpusnek_new_int("N", NUM_THREADS);
//     gpusnek_do_str(R"(

// from math import log

// cache[tid] = data[tid]
// syncthreads

// steps = int(log(N, 2))
// for i in range(steps):
//     offset = N >> (i + 1)
//     if tid < offset:
//         cache[tid] += cache[tid + offset]
//     syncthreads

// if tid == 0:
//     data[0] = cache[0] / len(data)
// )");
// }




// __global__
// void my_kernel(char* mem, int per_thread_mem) {
//     char* my_mem = &mem[threadIdx.x * per_thread_mem];
//     gpusnek_init(my_mem, per_thread_mem);

//     gpusnek_do_str("print('Hello World!')\n")
// }
