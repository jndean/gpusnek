#ifndef MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H
#define MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H

#include "py/mpconfig.h"
#include "py/parse.h"
#include "py/mpstate.h"
#include "py/runtime.h"



// Initialize the GPU Snek environment
MAYBE_CUDA void gpusnek_init(mp_state_ctx_t *ctx, void *memory, size_t stack_size, size_t heap_size);
MAYBE_CUDA void gpusnek_deinit(void);

// Bind device values / memory to be available in the python interpreter
// Supported typecodes:
// b, B, h, H, i, I, l, L, q, Q, f, d (see python's array module)
MAYBE_CUDA void gpusnek_bind_memory(const char *name, void *start, int len, char typecode);
MAYBE_CUDA void gpusnek_new_int(const char *name, int val);

MAYBE_CUDA void gpusnek_do_str(const char *src);
MAYBE_CUDA mp_obj_t gpusnek_compile(const char *src);
MAYBE_CUDA void gpusnek_do_bytecode(mp_obj_t compiled_module);
MAYBE_CUDA void gpusnek_do_mpy(const unsigned char *mpy, unsigned int mpy_len);

// Configure per-thread stdin/stdout buffers.
// Passing NULL disables buffered I/O for that stream and falls back to printf / -1.
// Re-calling resets the read/write head to 0 and NULL-terminates the buffer.
MAYBE_CUDA void gpusnek_set_stdin(char *address, int size);
MAYBE_CUDA void gpusnek_set_stdout(char *address, int size);
// Reset both read/write heads to 0 and NULL-terminate both buffers.
MAYBE_CUDA void gpusnek_reset_stdio_heads(void);


#endif // MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H
