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

// Execute a Python string
MAYBE_CUDA void gpusnek_do_str(const char *src, mp_parse_input_kind_t input_kind);


#endif // MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H
