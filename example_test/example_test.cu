// Main entry point for CUDA MicroPython port
// This is a minimal implementation for POC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpusnek/gpusnek.h"

// MicroPython memory configuration: per-thread allocation sizes
#define HEAP_SIZE (10 * 1024)
#define PYSTACK_SIZE (1 * 1024)

#define N_THREADS 2

MAYBE_CUDA void run_micropython_tests(void);

#ifdef __CUDACC__
void run_cuda_test(void);
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

    gpusnek_init(&host_state_ctx, memory_ptr, PYSTACK_SIZE, HEAP_SIZE);

    #if MICROPY_ENABLE_GC
    // Set the C stack top for conservative GC root scanning. This must be
    // done manually here because mp_init clears the state context.
    int stack_dummy;
    MP_STATE_THREAD(stack_top) = (char *)&stack_dummy;
    #endif

    run_micropython_tests();
    gpusnek_deinit();

    free(memory_ptr);
    #endif

    printf("CUDA MicroPython POC finished.\n");
    return 0;
}



MAYBE_CUDA void run_micropython_tests(void) {
    printf("Running MicroPython tests...\n");

    // Test 1: Basic arithmetic
    printf("Test 1: Basic arithmetic\n");
    gpusnek_do_str("print(1+2+3)", MP_PARSE_FILE_INPUT);

    // Test 2: Variables
    printf("Test 2: Variables\n");
    gpusnek_do_str("x = 42\nprint(x * 2)", MP_PARSE_FILE_INPUT);

    // Test 3: List comprehension
    printf("Test 3: List comprehension\n");
    gpusnek_do_str("squares = [x*x for x in range(5)]\nprint(squares)", MP_PARSE_FILE_INPUT);

    // Test 4: String formatting
    printf("Test 4: String formatting\n");
    gpusnek_do_str("name = 'CUDA'\nprint('Hello, {}!'.format(name))", MP_PARSE_FILE_INPUT);

    // Test 5: Class definition and method call
    printf("Test 5: Class definition\n");
    gpusnek_do_str(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "    def inc(self):\n"
        "        self.count += 1\n"
        "        return self.count\n"
        "c = Counter()\n"
        "print(c.inc(), c.inc(), c.inc())\n",
        MP_PARSE_FILE_INPUT);

    // Test 6: Monkey-patch a method
    printf("Test 6: Method patching\n");
    gpusnek_do_str(
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'Hello'\n"
        "def new_greet(self):\n"
        "    return 'Patched!'\n"
        "g = Greeter()\n"
        "Greeter.greet = new_greet\n"
        "print(g.greet())\n",
        MP_PARSE_FILE_INPUT);

    // Test 7: Lambda and higher-order functions
    printf("Test 7: Lambda and map\n");
    gpusnek_do_str("print(list(map(lambda x: x*2, [1,2,3])))", MP_PARSE_FILE_INPUT);

    // Test 8: Tuple unpacking
    printf("Test 8: Tuple unpacking\n");
    gpusnek_do_str("a, b, c = (10, 20, 30)\nprint(a + b + c)", MP_PARSE_FILE_INPUT);

    // Test 9: Dictionary
    printf("Test 9: Dictionary\n");
    gpusnek_do_str("d = {'a': 1, 'b': 2}\nprint(d['a'] + d['b'])", MP_PARSE_FILE_INPUT);

    // Test 10: Generator expression with sum
    printf("Test 10: Generator expression\n");
    gpusnek_do_str("print(sum(x for x in range(10)))", MP_PARSE_FILE_INPUT);

    // Test 11: Generator expression with sum
    printf("Test 11: Types\n");
    gpusnek_do_str("print(dir(type(type(1))))\n", MP_PARSE_FILE_INPUT);

    // Test 12: Per-thread __main__ module isolation
    // Each thread sets test_isolation to a DIFFERENT value (100 + thread_id),
    // then a separate gpusnek_do_str reads it back. If threads shared one __main__,
    // one thread would see the other's value.
    printf("Test 12: __main__ module isolation\n");
    {
        int tid = MP_THREAD_IDX;
        int val = 100 + tid;
        // Build "test_isolation = 1XX" — last digit varies by thread
        char set_src[] = "test_isolation = 100";
        set_src[19] = '0' + (val % 10);  // patch units digit
        set_src[18] = '0' + (val / 10) % 10;  // patch tens digit
        set_src[17] = '0' + (val / 100);  // patch hundreds digit
        gpusnek_do_str(set_src, MP_PARSE_FILE_INPUT);

        // Build "assert test_isolation == 1XX\nprint(test_isolation)"
        char chk_src[] = "assert test_isolation == 100\nprint(test_isolation)";
        chk_src[27] = '0' + (val % 10);
        chk_src[26] = '0' + (val / 10) % 10;
        chk_src[25] = '0' + (val / 100);
        gpusnek_do_str(chk_src, MP_PARSE_FILE_INPUT);
    }

    printf("Test 13: GC\n");
    gpusnek_do_str(
        "x = 1\n"
        "z = []\n"
        "for y in range(10000):\n"
        "    x += 1\n"
        "    y = [x, x+1, x+2]\n"
        "    z.append(y)\n"
        "    if len(z) > 6:\n"
        "        z.pop(0)\n"
        "print(z[-1])\n"
        , MP_PARSE_FILE_INPUT
    );

    // printf("Test 14: Exception\n");
    // gpusnek_do_str(
    //     "x = [1,2,3]\n"
    //     "print(x[10])\n",
    //     MP_PARSE_FILE_INPUT
    // );

    // Test 15: gpusnek_bind_memory — write to a C buffer from Python
    printf("Test 15: gpusnek_bind_memory\n");
    static unsigned char shared_buf[8] = {0};
    gpusnek_bind_memory("data", shared_buf, 8, 'B');
    gpusnek_do_str(
        "data[0] = 42\n"
        "data[7] = 255\n"
        "print(len(data), data[0], data[7])\n",
        MP_PARSE_FILE_INPUT
    );
    // Verify the writes actually landed in the C buffer
    printf("[C] shared_buf[0]=%d shared_buf[7]=%d\n",
           (int)shared_buf[0], (int)shared_buf[7]);

    // Test 16: syncthreads keyword
    printf("Test 16: syncthreads keyword\n");
    gpusnek_do_str(
        "syncthreads\n",
        MP_PARSE_FILE_INPUT
    );

    // Test 17: gpusnek_new_int
    printf("Test 17: gpusnek_new_int\n");
    gpusnek_new_int("my_config_val", 1337);
    gpusnek_do_str(
        "print('my_config_val is:', my_config_val)\n"
        "print('multiplied by 2:', my_config_val * 2)\n",
        MP_PARSE_FILE_INPUT
    );

    printf("MicroPython tests finished.\n");
}

#ifdef __CUDACC__

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
                                    char *memory_base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[%d] micropython_kernel started\n", tid);

    // Capture the stack pointer at kernel entry before ANY function calls.
    // Stack grows DOWN on CUDA: this is the highest (most-base) SP we'll ever
    // have.  After gpusnek_init's memset clears the state, we restore it so that
    // gc_collect sees a valid range [current_sp, entry_sp).
    void *kernel_entry_sp;
    asm("mov.u64 %0, %%SP;" : "=l"(kernel_entry_sp));

    // Each thread gets its own memory region (stack + heap)
    char *my_memory = memory_base + tid * (PYSTACK_SIZE + HEAP_SIZE);

    gpusnek_init(state_array, my_memory, PYSTACK_SIZE, HEAP_SIZE);

    // Restore stack_top to the kernel-entry SP, overriding what gpusnek_init
    // recorded (gpusnek_init's SP is lower = deeper in the stack).
    #if MICROPY_ENABLE_GC
    MP_STATE_THREAD(stack_top) = (char *)kernel_entry_sp;
    #endif

    run_micropython_tests();
    gpusnek_deinit();

    printf("[%d] micropython_kernel finished\n", tid);
    results[tid] = 42;
}

// Host function to launch the multi-thread test
void run_cuda_test(void) {
    printf("CUDA MicroPython multi-thread test starting (%d threads)...\n", N_THREADS);

    // Set GPU stack size (MicroPython needs deep stacks)
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 16*1024));
     
    // Allocate per-thread result array
    int *d_results;
    gpuErrchk(cudaMalloc(&d_results, N_THREADS * sizeof(int)));

    // Allocate per-thread state contexts
    mp_state_ctx_t *d_states;
    gpuErrchk(cudaMalloc(&d_states, N_THREADS * sizeof(mp_state_ctx_t)));

    // Allocate per-thread memory (contiguous block, each thread gets a slice for stack+heap)
    char *d_memory;
    gpuErrchk(cudaMalloc(&d_memory, N_THREADS * (PYSTACK_SIZE + HEAP_SIZE)));

    // Launch kernel with N threads
    micropython_kernel<<<1, N_THREADS>>>(d_results, d_states, d_memory);
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
    cudaFree(d_memory);
}

#endif // __CUDACC__
