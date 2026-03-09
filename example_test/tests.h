#ifndef MICROPY_INCLUDED_PORTS_CUDA_TESTS_H
#define MICROPY_INCLUDED_PORTS_CUDA_TESTS_H

#include "gpusnek/gpusnek_api.h"

// MicroPython memory configuration: per-thread allocation sizes
#define HEAP_SIZE (10 * 1024)
#define PYSTACK_SIZE (1 * 1024)

MAYBE_CUDA void run_micropython_tests(void);


#endif // MICROPY_INCLUDED_PORTS_CUDA_TESTS_H
