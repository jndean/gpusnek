#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpusnek/gpusnek.h"
#include "utils_for_examples.h"

#define HEAP_SIZE (5 * 1024)
#define PYSTACK_SIZE (1 * 1024)
#define PER_THREAD_MEMORY (PYSTACK_SIZE + HEAP_SIZE)

// Number of columns = number of threads
#define NUM_THREADS (1 << 16)

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
// Three compile-time bool toggles control which phases each launch performs.
// This lets us profile them individually.
template <bool DO_INIT, bool DO_COMPILE, bool DO_EXECUTE>
__global__ void sum_kernel_gpusnek(float *data, int num_rows, int num_threads,
                                    mp_state_ctx_t *states, char *memory,
                                    mp_obj_t *bytecode_slots) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    if (DO_INIT) {
        char *thread_mem = memory + tid * PER_THREAD_MEMORY;
        gpusnek_init(states, thread_mem, PYSTACK_SIZE, HEAP_SIZE);
    }

    if (DO_COMPILE) {
        gpusnek_bind_memory("data", data, num_rows * num_threads, 'f');
        gpusnek_new_int("tid", tid);
        gpusnek_new_int("num_rows", num_rows);
        gpusnek_new_int("num_threads", num_threads);
        mp_obj_t bc = gpusnek_compile(R"(
acc = 0.0
for row in range(num_rows):
    acc += data[row * num_threads + tid]
data[tid] = acc
)");
        bytecode_slots[tid] = bc;
    }

    if (DO_EXECUTE) {
        gpusnek_do_bytecode(bytecode_slots[tid]);
    }
}


static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--num_rows N] [--mode MODE]\n"
        "\n"
        "  --num_rows N   Number of rows to sum (default 256)\n"
        "  --mode M       0 = baseline    (pure CUDA C kernel)\n"
        "                 1 = basic       (init+compile+run in one launch)\n"
        "                 2 = preinit     (launch1: init, launch2: compile+run)\n"
        "                 3 = precompile  (launch1: init+compile, launch2: run)\n",
        prog);
}

int main(int argc, char **argv) {
    long num_rows = 256;
    int mode = 1;

    // --- Parse command-line arguments ---
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--num_rows") == 0 && i + 1 < argc) {
            num_rows = atol(argv[++i]);
            if (num_rows <= 0) {
                fprintf(stderr, "Invalid --num_rows value: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = atoi(argv[++i]);
            if (mode < 0 || mode > 3) {
                fprintf(stderr, "Invalid --mode value: %s (must be 0, 1, 2, or 3)\n", argv[i]);
                return 1;
            }
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const char *mode_names[] = { "baseline", "basic", "preinit", "precompile" };
    long total_elements = (long)NUM_THREADS * num_rows;
    printf("Sum Profiling: %d threads x %ld rows = %ld elements, mode=%d (%s)\n",
           NUM_THREADS, num_rows, total_elements, mode, mode_names[mode]);

    // --- Allocate and fill host data ---
    size_t data_bytes = total_elements * sizeof(float);
    float *h_data = (float *)malloc(data_bytes);
    for (long i = 0; i < total_elements; i++) {
        h_data[i] = 1.0f;  // every element is 1, so each column sum = num_rows
    }

    // --- Allocate device memory for data ---
    float *d_data;
    catchError(cudaMalloc(&d_data, data_bytes));
    catchError(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    // --- Launch config ---
    int threads_per_block = 256;
    int num_blocks = (NUM_THREADS + threads_per_block - 1) / threads_per_block;

    // --- Gpusnek state (only allocated for modes 1-3) ---
    mp_state_ctx_t *d_states = NULL;
    char *d_memory = NULL;
    mp_obj_t *d_bytecode_slots = NULL;

    if (mode >= 1) {
        catchError(cudaMalloc(&d_states, (size_t)NUM_THREADS * sizeof(mp_state_ctx_t)));
        catchError(cudaMalloc(&d_memory, (size_t)NUM_THREADS * PER_THREAD_MEMORY));
        catchError(cudaMalloc(&d_bytecode_slots, (size_t)NUM_THREADS * sizeof(mp_obj_t)));
        catchError(cudaDeviceSetLimit(cudaLimitStackSize, 5 * 1024));
    }

    // --- Launch kernel(s) ---
    if (mode == 0) {
        // baseline: pure CUDA C kernel
        sum_kernel_cuda<<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());

    } else if (mode == 1) {
        // basic: single launch does init + compile + run
        sum_kernel_gpusnek<true, true, true><<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS, d_states, d_memory, d_bytecode_slots);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());

    } else if (mode == 2) {
        // preinit: launch 1 does init only
        sum_kernel_gpusnek<true, false, false><<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS, d_states, d_memory, d_bytecode_slots);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());
        // launch 2 does compile + run
        sum_kernel_gpusnek<false, true, true><<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS, d_states, d_memory, d_bytecode_slots);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());

    } else /* mode == 3 */ {
        // precompile: launch 1 does init + compile
        sum_kernel_gpusnek<true, true, false><<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS, d_states, d_memory, d_bytecode_slots);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());
        // launch 2 does run only
        sum_kernel_gpusnek<false, false, true><<<num_blocks, threads_per_block>>>(
            d_data, (int)num_rows, NUM_THREADS, d_states, d_memory, d_bytecode_slots);
        catchError(cudaGetLastError());
        catchError(cudaDeviceSynchronize());
    }

    printf("Done.\n");

    // --- Verify results ---
    float *h_result = (float *)malloc(NUM_THREADS * sizeof(float));
    catchError(cudaMemcpy(h_result, d_data,
                          NUM_THREADS * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = (float)num_rows;
    int pass = 1;
    for (int i = 0; i < NUM_THREADS; i++) {
        if (h_result[i] != expected) { pass = 0; break; }
    }

    printf("Result: %s (expected %.0f)\n", pass ? "PASS" : "FAIL", expected);

    // --- Cleanup ---
    cudaFree(d_data);
    if (d_states) cudaFree(d_states);
    if (d_memory) cudaFree(d_memory);
    if (d_bytecode_slots) cudaFree(d_bytecode_slots);
    free(h_data);
    free(h_result);
    return 0;
}

#else // !__CUDACC__

int main(void) {
    printf("example_sum_for_profiling: CUDA-only example, not supported on host.\n");
    return 0;
}

#endif // __CUDACC__
