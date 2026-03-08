#ifndef MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H
#define MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H

#include "py/mpconfig.h"
#include "py/parse.h"
#include "py/runtime.h"


// Per-thread heap size for the bump allocator
#define BUMP_ALLOC_HEAP_SIZE (10 * 1024)
#define PYSTACK_SIZE (1 * 1024)

MAYBE_CUDA void do_str(const char *src, mp_parse_input_kind_t input_kind);
MAYBE_CUDA void mp_bind_array(const char *name, void *start, int len);


#endif // MICROPY_INCLUDED_PORTS_CUDA_GPUSNEK_API_H
