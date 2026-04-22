#include <stdio.h>
#include <stdlib.h>

#include "gpusnek/gpusnek.h"
#include "utils_for_examples.h"

#define HEAP_SIZE (5 * 1024)
#define PYSTACK_SIZE (1 * 1024)
#define PER_THREAD_MEMORY (PYSTACK_SIZE + HEAP_SIZE)

// Number of columns = number of threads
#define NUM_THREADS (1 << 16)

// Number of rows — override via environment variable, default 256
#ifndef NUM_ROWS
#define NUM_ROWS 256
#endif

#ifdef __CUDACC__

// ---------- Pure CUDA C kernel ----------
__global__ void sum_kernel_cuda(float *data, int num_rows, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    float acc = 0.0f;
    for (int row = 0; row < num_rows; row++) {
        acc += data[row * num_threads + tid];
    }
    data[tid] = acc;  // write sum into row 0
}

// ---------- Gpusnek kernel ----------
__global__ void sum_kernel_gpusnek(float *data, int num_rows, int num_threads,
                                    mp_state_ctx_t *states, char *memory) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    char *thread_mem = memory + tid * PER_THREAD_MEMORY;
    gpusnek_init(states, thread_mem, PYSTACK_SIZE, HEAP_SIZE);

    gpusnek_bind_memory("data", data, num_rows * num_threads, 'f');
    gpusnek_new_int("tid", tid);
    gpusnek_new_int("num_rows", num_rows);
    gpusnek_new_int("num_threads", num_threads);
    gpusnek_do_str(R"(
acc = 0.0
for row in range(num_rows):
    acc += data[row * num_threads + tid]
data[tid] = acc
)");
}


int main(void) {
    long num_rows = NUM_ROWS;

    // Allow overriding NUM_ROWS via environment variable
    const char *env_rows = getenv("NUM_ROWS");
    if (env_rows) {
        num_rows = atol(env_rows);
        if (num_rows <= 0) {
            fprintf(stderr, "Invalid NUM_ROWS value: %s\n", env_rows);
            return 1;
        }
    }

    long total_elements = (long)NUM_THREADS * num_rows;
    printf("Sum Comparison: %d threads x %ld rows = %ld elements\n",
           NUM_THREADS, num_rows, total_elements);

    // --- Allocate and fill host data ---
    size_t data_bytes = total_elements * sizeof(float);
    float *h_data = (float *)malloc(data_bytes);
    for (long i = 0; i < total_elements; i++) {
        h_data[i] = 1.0f;  // every element is 1, so each column sum = num_rows
    }

    // --- Allocate device memory for data ---
    float *d_data_cuda, *d_data_snek;
    catchError(cudaMalloc(&d_data_cuda, data_bytes));
    catchError(cudaMalloc(&d_data_snek, data_bytes));
    catchError(cudaMemcpy(d_data_cuda, h_data, data_bytes, cudaMemcpyHostToDevice));
    catchError(cudaMemcpy(d_data_snek, h_data, data_bytes, cudaMemcpyHostToDevice));

    // --- Allocate gpusnek state memory ---
    mp_state_ctx_t *d_states;
    char *d_memory;
    catchError(cudaMalloc(&d_states, (size_t)NUM_THREADS * sizeof(mp_state_ctx_t)));
    catchError(cudaMalloc(&d_memory, (size_t)NUM_THREADS * PER_THREAD_MEMORY));
    catchError(cudaDeviceSetLimit(cudaLimitStackSize, 5 * 1024));

    // --- Launch config ---
    int threads_per_block = 256;
    int num_blocks = (NUM_THREADS + threads_per_block - 1) / threads_per_block;

    // --- Launch pure CUDA kernel ---
    printf("Launching CUDA C kernel...\n");
    sum_kernel_cuda<<<num_blocks, threads_per_block>>>(
        d_data_cuda, (int)num_rows, NUM_THREADS);
    catchError(cudaGetLastError());
    catchError(cudaDeviceSynchronize());
    printf("CUDA C kernel done.\n");

    // --- Launch gpusnek kernel ---
    printf("Launching gpusnek kernel...\n");
    sum_kernel_gpusnek<<<num_blocks, threads_per_block>>>(
        d_data_snek, (int)num_rows, NUM_THREADS, d_states, d_memory);
    catchError(cudaGetLastError());
    catchError(cudaDeviceSynchronize());
    printf("Gpusnek kernel done.\n");

    // --- Verify results (check all columns from row 0) ---
    float *h_result_cuda = (float *)malloc(NUM_THREADS * sizeof(float));
    float *h_result_snek = (float *)malloc(NUM_THREADS * sizeof(float));
    catchError(cudaMemcpy(h_result_cuda, d_data_cuda,
                          NUM_THREADS * sizeof(float), cudaMemcpyDeviceToHost));
    catchError(cudaMemcpy(h_result_snek, d_data_snek,
                          NUM_THREADS * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = (float)num_rows;
    int pass_cuda = 1, pass_snek = 1;
    for (int i = 0; i < NUM_THREADS; i++) {
        if (h_result_cuda[i] != expected) { pass_cuda = 0; break; }
        if (h_result_snek[i] != expected) { pass_snek = 0; break; }
    }

    printf("CUDA C kernel:   %s (expected %.0f)\n",
           pass_cuda ? "PASS" : "FAIL", expected);
    printf("Gpusnek kernel:  %s (expected %.0f)\n",
           pass_snek ? "PASS" : "FAIL", expected);

    // --- Cleanup ---
    cudaFree(d_data_cuda);
    cudaFree(d_data_snek);
    cudaFree(d_states);
    cudaFree(d_memory);
    free(h_data);
    free(h_result_cuda);
    free(h_result_snek);
    return 0;
}

#else // !__CUDACC__

int main(void) {
    printf("example_sum_comparison: CUDA-only example, not supported on host.\n");
    return 0;
}

#endif // __CUDACC__
