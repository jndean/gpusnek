/*
 * Bump allocator for CUDA device code.
 *
 * This provides a simple bump allocator that replaces malloc/realloc/free
 * when running on the GPU. The allocator works by advancing a pointer
 * through a pre-allocated buffer. Free is a no-op.
 *
 * Usage:
 *   Call bump_alloc_init(heap_ptr, heap_size) before using.
 */

#ifndef MICROPY_BUMPALLOC_H
#define MICROPY_BUMPALLOC_H

#include <stdint.h>
#include <stddef.h>
#include "py/mpconfig.h"


// API
MAYBE_CUDA void bump_alloc_init(char *heap_ptr, size_t heap_size);
MAYBE_CUDA void *bump_malloc(size_t num_bytes);
MAYBE_CUDA void *bump_realloc(void *ptr, size_t new_num_bytes);
MAYBE_CUDA void bump_free(void *ptr);

#endif // MICROPY_BUMPALLOC_H
