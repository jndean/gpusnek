/*
 * Bump allocator implementation.
 */

#include <stdio.h>
#include <string.h>
#include "py/bumpalloc.h"

// Alignment for all allocations (8 bytes for 64-bit alignment)
#define BUMP_ALLOC_ALIGN 8

// Globals
MAYBE_CUDA static char *bump_heap_base = 0;
MAYBE_CUDA static size_t bump_heap_offset = 0;
MAYBE_CUDA static size_t bump_heap_size = 0;

// Init function
MAYBE_CUDA void bump_alloc_init(char *heap_ptr, size_t heap_size) {
    bump_heap_base = heap_ptr;
    bump_heap_offset = 0;
    bump_heap_size = heap_size;
}

MAYBE_CUDA void *bump_malloc(size_t num_bytes) {
    if (num_bytes == 0) {
        return NULL;
    }
    // Align the size up
    size_t aligned = (num_bytes + BUMP_ALLOC_ALIGN - 1) & ~(BUMP_ALLOC_ALIGN - 1);

    size_t old_offset;
    #ifdef __CUDA_ARCH__
    // Atomicly advance the offset on device
    old_offset = atomicAdd((unsigned long long *)&bump_heap_offset, (unsigned long long)aligned);
    #else
    // Normal increment on host
    old_offset = bump_heap_offset;
    bump_heap_offset += aligned;
    #endif

    if (old_offset + aligned > bump_heap_size) {
        // Out of memory - return NULL
        printf("OOM\n");
        return NULL;
    }

    return (void *)(bump_heap_base + old_offset);
}

MAYBE_CUDA void *bump_realloc(void *ptr, size_t new_num_bytes) {
    // Simple strategy: allocate new block and copy
    if (ptr == NULL) {
        return bump_malloc(new_num_bytes);
    }
    if (new_num_bytes == 0) {
        return NULL;
    }
    void *new_ptr = bump_malloc(new_num_bytes);
    if (new_ptr != NULL) {
        // Copy the data. We don't know the old size, so we copy new_num_bytes.
        char *dst = (char *)new_ptr;
        char *src = (char *)ptr;
        for (size_t i = 0; i < new_num_bytes; i++) {
            dst[i] = src[i];
        }
    }
    return new_ptr;
}

MAYBE_CUDA void bump_free(void *ptr) {
    // Bump allocator: free is a no-op
    (void)ptr;
}
