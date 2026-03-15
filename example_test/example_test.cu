// Main entry point for CUDA MicroPython port
// This is a minimal implementation for POC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpusnek/gpusnek.h"

// MicroPython memory configuration: per-thread allocation sizes
#define PYSTACK_SIZE  (1 * 1024)
#define HEAP_SIZE     (10 * 1024)
#define STDIN_SIZE    (512)
#define STDOUT_SIZE   (2 * 1024)
#define MEM_PER_THREAD (PYSTACK_SIZE + HEAP_SIZE + STDIN_SIZE + STDOUT_SIZE)
// Layout per thread: [pystack | heap | stdin | stdout]
#define HEAP_OFFSET (PYSTACK_SIZE)
#define STDIN_OFFSET (HEAP_OFFSET + HEAP_SIZE)
#define STDOUT_OFFSET (STDIN_OFFSET + STDIN_SIZE)

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
    // Host build: single allocation block: [pystack | heap | stdin | stdout]
    static mp_state_ctx_t host_state_ctx;
    char *memory_ptr = (char *)malloc(MEM_PER_THREAD);
    if (!memory_ptr) {
        printf("FATAL: Failed to allocate memory\n");
        return 1;
    }

    gpusnek_init(&host_state_ctx, memory_ptr, PYSTACK_SIZE, HEAP_SIZE);
    gpusnek_set_stdin (memory_ptr + STDIN_OFFSET,  STDIN_SIZE);
    gpusnek_set_stdout(memory_ptr + STDOUT_OFFSET, STDOUT_SIZE);

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

    // Macro: reset stdout head, run test body, then print what Python wrote.
    #define TEST(label, body) \
        do { \
            printf(label "\n"); \
            gpusnek_reset_stdio_heads(); \
            body \
            printf("%s", MP_STATE_CTX.io.stdout_buf); \
        } while(0)

    // Test 1: Basic arithmetic
    TEST("Test 1: Basic arithmetic",
        gpusnek_do_str("print(1+2+3)");
    );

    // Test 2: Variables
    TEST("Test 2: Variables",
        gpusnek_do_str("x = 42\nprint(x * 2)");
    );

    // Test 3: List comprehension
    TEST("Test 3: List comprehension",
        gpusnek_do_str("squares = [x*x for x in range(5)]\nprint(squares)");
    );

    // Test 4: String formatting
    TEST("Test 4: String formatting",
        gpusnek_do_str("name = 'CUDA'\nprint('Hello, {}!'.format(name))");
    );

    // Test 5: Class definition and method call
    TEST("Test 5: Class definition",
        gpusnek_do_str(
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "    def inc(self):\n"
            "        self.count += 1\n"
            "        return self.count\n"
            "c = Counter()\n"
            "print(c.inc(), c.inc(), c.inc())\n"
        );
    );

    // Test 6: Monkey-patch a method
    TEST("Test 6: Method patching",
        gpusnek_do_str(
            "class Greeter:\n"
            "    def greet(self):\n"
            "        return 'Hello'\n"
            "def new_greet(self):\n"
            "    return 'Patched!'\n"
            "g = Greeter()\n"
            "Greeter.greet = new_greet\n"
            "print(g.greet())\n"
        );
    );

    // Test 7: Lambda and higher-order functions
    TEST("Test 7: Lambda and map",
        gpusnek_do_str("print(list(map(lambda x: x*2, [1,2,3])))");
    );

    // Test 8: Tuple unpacking
    TEST("Test 8: Tuple unpacking",
        gpusnek_do_str("a, b, c = (10, 20, 30)\nprint(a + b + c)");
    );

    // Test 9: Dictionary
    TEST("Test 9: Dictionary",
        gpusnek_do_str("d = {'a': 1, 'b': 2}\nprint(d['a'] + d['b'])");
    );

    // Test 10: Generator expression with sum
    TEST("Test 10: Generator expression",
        gpusnek_do_str("print(sum(x for x in range(10)))");
    );

    // Test 11: Types
    TEST("Test 11: Types",
        gpusnek_do_str("print(dir(type(type(1))))\n");
    );

    // Test 12: Per-thread __main__ module isolation
    TEST("Test 12: __main__ module isolation",
        {
            int tid = MP_THREAD_IDX;
            int val = 100 + tid;
            char set_src[] = "test_isolation = 100";
            set_src[19] = '0' + (val % 10);
            set_src[18] = '0' + (val / 10) % 10;
            set_src[17] = '0' + (val / 100);
            gpusnek_do_str(set_src);
            char chk_src[] = "assert test_isolation == 100\nprint(test_isolation)";
            chk_src[27] = '0' + (val % 10);
            chk_src[26] = '0' + (val / 10) % 10;
            chk_src[25] = '0' + (val / 100);
            gpusnek_do_str(chk_src);
        }
    );

    // Test 13: GC
    TEST("Test 13: GC",
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
        );
    );

    // Test 14: fstrings
    TEST("Test 14: fstrings",
        gpusnek_do_str(
            "print(f'help{\"me\"}')\n"
        );
    );

    // Test 15: gpusnek_bind_memory — write to a C buffer from Python
    TEST("Test 15: gpusnek_bind_memory",
        static unsigned char shared_buf[8] = {0};
        gpusnek_bind_memory("data", shared_buf, 8, 'B');
        gpusnek_do_str(
            "data[0] = 42\n"
            "data[7] = 255\n"
            "print(len(data), data[0], data[7])\n"
        );
        printf("[C] shared_buf[0]=%d shared_buf[7]=%d\n",
               (int)shared_buf[0], (int)shared_buf[7]);
    );

    // Test 16: syncthreads keyword
    TEST("Test 16: syncthreads keyword",
        gpusnek_do_str("syncthreads\n");
    );

    // Test 17: gpusnek_new_int
    TEST("Test 17: gpusnek_new_int",
        gpusnek_new_int("my_config_val", 1337);
        gpusnek_do_str(
            "print('my_config_val is:', my_config_val)\n"
            "print('multiplied by 2:', my_config_val * 2)\n"
        );
    );

    #undef TEST

    printf("MicroPython tests finished.\n");
}

// Helper: reset the stdout buffer and run a named test, then printf its output.
#define RUN_TEST(label, body) do { \
    gpusnek_reset_stdio_heads(); \
    body \
    printf("%s", MP_STATE_CTX.io.stdout_buf); \
} while(0)

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

    // Each thread gets its own memory region: [pystack | heap | stdin | stdout]
    char *my_memory = memory_base + tid * MEM_PER_THREAD;

    gpusnek_init(state_array, my_memory, PYSTACK_SIZE, HEAP_SIZE);
    gpusnek_set_stdin (my_memory + STDIN_OFFSET,  STDIN_SIZE);
    gpusnek_set_stdout(my_memory + STDOUT_OFFSET, STDOUT_SIZE);

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

    // Single allocation: each thread gets MEM_PER_THREAD bytes (pystack+heap+stdin+stdout)
    char *d_memory;
    gpuErrchk(cudaMalloc(&d_memory, N_THREADS * MEM_PER_THREAD));

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
