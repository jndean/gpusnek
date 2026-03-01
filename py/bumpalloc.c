/*
 * Bump allocator implementation.
 *
 * Each thread's state lives inside mp_state_ctx_t.bump, accessed via
 * MP_STATE_CTX.bump. Functions cache the state pointer locally to
 * avoid repeated index lookups.
 */

#include <stdio.h>
#include <string.h>
#include "py/bumpalloc.h"
#include "py/mpstate.h"

// Alignment for all allocations (8 bytes for 64-bit alignment)
#define BUMP_ALLOC_ALIGN 8

// Init function — initializes current thread's allocator state
MAYBE_CUDA void bump_alloc_init(char *heap_ptr, size_t heap_size) {
    bump_alloc_state_t *s = &MP_STATE_CTX.bump;
    s->base = heap_ptr;
    s->offset = 0;
    s->size = heap_size;
}

MAYBE_CUDA void *bump_malloc(size_t num_bytes) {
    if (num_bytes == 0) {
        return NULL;
    }

    bump_alloc_state_t *s = &MP_STATE_CTX.bump;

    // Align the size up
    size_t aligned = (num_bytes + BUMP_ALLOC_ALIGN - 1) & ~(BUMP_ALLOC_ALIGN - 1);

    size_t old_offset = s->offset;
    s->offset += aligned;

    if (old_offset + aligned > s->size) {
        printf("OOM\n");
        return NULL;
    }

    return (void *)(s->base + old_offset);
}

MAYBE_CUDA void *bump_realloc(void *ptr, size_t new_num_bytes) {
    if (ptr == NULL) {
        return bump_malloc(new_num_bytes);
    }
    if (new_num_bytes == 0) {
        return NULL;
    }
    void *new_ptr = bump_malloc(new_num_bytes);
    if (new_ptr != NULL) {
        char *dst = (char *)new_ptr;
        char *src = (char *)ptr;
        for (size_t i = 0; i < new_num_bytes; i++) {
            dst[i] = src[i];
        }
    }
    return new_ptr;
}

MAYBE_CUDA void bump_free(void *ptr) {
    (void)ptr;
}
