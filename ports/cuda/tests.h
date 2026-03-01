#ifndef MICROPY_INCLUDED_PORTS_CUDA_TESTS_H
#define MICROPY_INCLUDED_PORTS_CUDA_TESTS_H

#include "py/mpconfig.h"

// Per-thread heap size for the bump allocator
#define BUMP_ALLOC_HEAP_SIZE (100 * 1024)

MAYBE_CUDA void do_str(const char *src, mp_parse_input_kind_t input_kind);
MAYBE_CUDA void run_micropython_tests(void);

#endif // MICROPY_INCLUDED_PORTS_CUDA_TESTS_H
