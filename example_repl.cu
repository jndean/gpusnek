// example_repl.cu — Interactive GPU MicroPython REPL
//
// Architecture:
//   1. setup_kernel (32 threads): allocates per-thread memory, runs gpusnek_init,
//      binds 'tid' variable, gpusnek_set_stdout, pyexec_event_repl_init to start the friendly REPL.
//   2. repl_kernel (32 threads, called per-keystroke): resets the stdout head,
//      feeds one character to pyexec_event_repl_process_char, returns done flag.
//   3. Host loop: puts terminal in raw mode, reads one char at a time, launches
//      repl_kernel, copies memory back, deduplicates output from all 32 threads,
//      and prints it.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <map>
#include <string>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "gpusnek/gpusnek.h"
#include "shared/runtime/pyexec.h"   // pyexec_event_repl_init, pyexec_event_repl_process_char

#define N_THREADS_PER_BLOCK (256)
#define N_BLOCKS            ((long)4096)
#define N_THREADS           (N_BLOCKS * N_THREADS_PER_BLOCK)
#define MAX_PRINTED_OUTPUTS 256
// Memory layout per interpreter
#define PYSTACK_SIZE   (1 * 1024)
#define HEAP_SIZE      (12 * 1024)
#define STDOUT_SIZE    (256)
#define HEAP_OFFSET    (PYSTACK_SIZE)
#define STDOUT_OFFSET  (HEAP_OFFSET + HEAP_SIZE)
#define MEM_PER_THREAD (PYSTACK_SIZE + HEAP_SIZE + STDOUT_SIZE)


// setup_kernel: called once to initialise the MicroPython interpreters.
// Writes the MicroPython banner to the stdout buffer so the host can display it.
__global__ void setup_kernel(char *memory, mp_state_ctx_t *states) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)N_THREADS) return;

    char *thread_mem = memory + tid * (size_t)MEM_PER_THREAD;
    gpusnek_init(states, thread_mem, PYSTACK_SIZE, HEAP_SIZE);
    gpusnek_set_stdout(thread_mem + STDOUT_OFFSET, STDOUT_SIZE);

    // Record the Python stack top from inside the kernel so GC root scanning
    // works correctly on the GPU stack.
#if MICROPY_ENABLE_GC
    volatile int stack_anchor = 0;
    MP_STATE_THREAD(stack_top) = (char *)&stack_anchor;
#endif

    // Bind 'threadIdx' into each thread's global namespace to allow thread divergence
    gpusnek_new_int("threadIdx", tid);

    // Initialise the event-driven friendly REPL — this prints the banner and
    // first prompt into the stdout buffer.
    pyexec_event_repl_init();
}

// repl_kernel: feed one character to the REPL and capture any output.
// Returns non-zero in *done if the REPL requested a soft-reset (CTRL-D).
__global__ void repl_kernel(int c, int *done) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)N_THREADS) return;

    #if MICROPY_ENABLE_GC
    volatile int stack_anchor = 0;
    MP_STATE_THREAD(stack_top) = (char *)&stack_anchor;
    #endif
    
    // Reset the stdout write head so we only capture new output from this char
    gpusnek_reset_stdio_heads();
    
    *done = 1;
    int ret = pyexec_event_repl_process_char(c);
    if (!(ret & PYEXEC_FORCED_EXIT)) *done = 0;
}

//  Terminal raw-mode helpers

static struct termios g_orig_termios;

static void restore_termios(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
}

static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    atexit(restore_termios);
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);   // no echo, char-at-a-time, no Ctrl-C signal
    raw.c_iflag &= ~(ICRNL | IXON);           // no CR→NL, no Ctrl-S/Q
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void extract_and_print_output(const char *h_memory) {
    std::map<std::string, int> output_counts;
    
    for (size_t i = 0; i < (size_t)N_THREADS; i++) {
        const char *stdout_buf = h_memory + i * (size_t)STDOUT_SIZE;
        // Make absolutely sure it's null-terminated by ignoring anything past STDOUT_SIZE.
        // gpusnek_set_stdout safely writes a null byte at pos 0, and hal_stdout updates it, 
        // but just to be safe in C++ mapping:
        std::string s(stdout_buf, strnlen(stdout_buf, STDOUT_SIZE));
        if (!s.empty()) {
            output_counts[s]++;
        }
    }

    if (output_counts.size() == 1) {
        printf("%s", output_counts.begin()->first.c_str());
    } else if (output_counts.size() > 1) {
        printf("\r\n");
        std::string common_prompt;
        const char *prompts[] = {"\r\n>>> ", "\r\n... ", ">>> ", "... "};
        
        for (const char *p : prompts) {
            std::string prompt_str(p);
            size_t plen = prompt_str.length();
            bool all_end_with_p = true;
            for (const auto& pair : output_counts) {
                if (pair.first.length() < plen || pair.first.substr(pair.first.length() - plen) != p) {
                    all_end_with_p = false;
                    break;
                }
            }
            if (all_end_with_p) {
                common_prompt = p;
                break;
            }
        }

        int print_count = 0;
        for (const auto& pair : output_counts) {
            if (print_count >= MAX_PRINTED_OUTPUTS) {
                printf("\r\n\x1b[1;31m------- WARNING: number of unique outputs exceeds cap, only printed the first %d -------\x1b[0m\r\n", MAX_PRINTED_OUTPUTS);
                break;
            }
            printf("\x1b[1;36m------- %dx -------\x1b[0m", pair.second);
            if (!common_prompt.empty()) {
                std::string stripped = pair.first.substr(0, pair.first.length() - common_prompt.length());
                printf("%s", stripped.c_str());
            } else {
                printf("%s", pair.first.c_str());
            }
            print_count++;
        }
        
        if (!common_prompt.empty()) {
            printf("%s", common_prompt.c_str());
        }
    }
    fflush(stdout);
}

int main(void) {
    // Allow enough stack for deep MicroPython recursion
    cudaDeviceSetLimit(cudaLimitStackSize, 8 * 1024);

    char *d_memory = NULL;
    mp_state_ctx_t *d_states = NULL;
    int *d_done = NULL;
    char *h_memory = NULL;
    int ret_code = 0;
    int done = 0;
    cudaError_t err;

    printf("Allocating %0.2f GB for on-device interpreter memory\n", (double)N_THREADS * MEM_PER_THREAD / (1 << 30));
    if (cudaMalloc((void **)&d_memory, (size_t)N_THREADS * MEM_PER_THREAD) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for memory failed\n");
        goto error;
    }
    
    if (cudaMalloc((void **)&d_states, (size_t)N_THREADS * sizeof(mp_state_ctx_t)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for states failed\n");
        goto error;
    }

    if (cudaMalloc((void **)&d_done, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for done flag failed\n");
        goto error;
    }

    h_memory = (char *)malloc((size_t)N_THREADS * STDOUT_SIZE);
    if (!h_memory) {
        fprintf(stderr, "malloc for h_memory failed\n");
        goto error;
    }

    setup_kernel<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(d_memory, d_states);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "setup_kernel failed: %s\n", cudaGetErrorString(err));
        goto error;
    }

    cudaMemcpy2D(h_memory, STDOUT_SIZE, d_memory + STDOUT_OFFSET, MEM_PER_THREAD, STDOUT_SIZE, N_THREADS, cudaMemcpyDeviceToHost);
    extract_and_print_output(h_memory);

    enable_raw_mode();

    while (!done) {
        // Read exactly one byte from the host terminal
        char raw_c;
        ssize_t n = read(STDIN_FILENO, &raw_c, 1);
        if (n <= 0) break;  // EOF / error

        int c = (unsigned char)raw_c;

        // Send byte to GPU REPL
        repl_kernel<<<N_BLOCKS, N_THREADS_PER_BLOCK>>>(c, d_done);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            // Restore terminal before printing error
            restore_termios();
            fprintf(stderr, "\nrepl_kernel error: %s\n", cudaGetErrorString(err));
            goto error;
        }

        // Read back any output the REPL produced and print it
        cudaMemcpy2D(h_memory, STDOUT_SIZE, d_memory + STDOUT_OFFSET, MEM_PER_THREAD, STDOUT_SIZE, N_THREADS, cudaMemcpyDeviceToHost);
        extract_and_print_output(h_memory);

        // Check if the REPL signalled exit (Ctrl-D soft-reset)
        int h_done = 0;
        cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_done) {
            done = 1;
        }
    }

    restore_termios();
    printf("\n[REPL exited]\n");
    goto cleanup;

error:
    ret_code = 1;
cleanup:
    if (h_memory) free(h_memory);
    if (d_memory) cudaFree(d_memory);
    if (d_states) cudaFree(d_states);
    if (d_done)   cudaFree(d_done);
    
    return ret_code;
}
