/*
 * Bump allocator for MicroPython.
 *
 * This provides a simple bump allocator that replaces malloc/realloc/free
 * when running on the GPU. The allocator works by advancing a pointer
 * through a pre-allocated buffer. Free is a no-op.
 *
 * Each thread's state lives inside mp_state_ctx_t.bump and is
 * automatically per-thread via MP_STATE_CTX.
 *
 * The caller allocates a heap buffer and passes it to mp_init(),
 * which calls bump_alloc_init() internally.
 */

#ifndef MICROPY_BUMPALLOC_H
#define MICROPY_BUMPALLOC_H

#include <stdint.h>
#include <stddef.h>
#include "py/mpconfig.h"

// Per-thread bump allocator state
typedef struct {
    char *base;
    size_t offset;
    size_t size;
} bump_alloc_state_t;

// API (called internally by mp_init, or by the allocator itself)
MAYBE_CUDA void bump_alloc_init(char *heap_ptr, size_t heap_size);
MAYBE_CUDA void *bump_malloc(size_t num_bytes);
MAYBE_CUDA void *bump_realloc(void *ptr, size_t new_num_bytes);
MAYBE_CUDA void bump_free(void *ptr);

#endif // MICROPY_BUMPALLOC_H
