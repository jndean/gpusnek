/*
 * Bump allocator for CUDA device code.
 *
 * This provides a simple bump allocator that replaces malloc/realloc/free
 * when running on the GPU. The allocator works by advancing a pointer
 * through a pre-allocated buffer. Free is a no-op.
 *
 * Each thread gets its own allocator state via bump_alloc_states array,
 * indexed by MP_THREAD_IDX.
 *
 * Usage:
 *   Set bump_alloc_states to point at an array of bump_alloc_state_t.
 *   Call bump_alloc_init(heap_ptr, heap_size) to initialize current thread's state.
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

// Array of per-thread states, set by launcher before bump_alloc_init
extern MAYBE_CUDA bump_alloc_state_t *bump_alloc_states;

// API
MAYBE_CUDA void bump_alloc_init(char *heap_ptr, size_t heap_size);
MAYBE_CUDA void *bump_malloc(size_t num_bytes);
MAYBE_CUDA void *bump_realloc(void *ptr, size_t new_num_bytes);
MAYBE_CUDA void bump_free(void *ptr);

#endif // MICROPY_BUMPALLOC_H
