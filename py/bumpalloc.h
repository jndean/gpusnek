/*
 * Bump allocator for CUDA device code.
 *
 * This provides a simple bump allocator that replaces malloc/realloc/free
 * when running on the GPU. The allocator works by advancing a pointer
 * through a pre-allocated buffer. Free is a no-op.
 *
 * Usage:
 *   Host side: call bump_alloc_init() before launching any kernels
 *   Device side: m_malloc/m_realloc/m_free use this automatically
 */

#ifndef MICROPY_BUMPALLOC_H
#define MICROPY_BUMPALLOC_H

#include <stdint.h>
#include <stddef.h>

// Default bump allocator heap size (256KB)
#ifndef BUMP_ALLOC_HEAP_SIZE
#define BUMP_ALLOC_HEAP_SIZE (256 * 1024)
#endif

// Alignment for all allocations (8 bytes for 64-bit alignment)
#define BUMP_ALLOC_ALIGN 8

#ifdef __CUDA_ARCH__

// Device-side globals: pointer to heap buffer and current offset
extern __device__ char *bump_heap_base;
extern __device__ size_t bump_heap_offset;
extern __device__ size_t bump_heap_size;

static __device__ inline void *bump_malloc(size_t num_bytes) {
    if (num_bytes == 0) {
        return (void *)0;
    }
    // Align the size up
    size_t aligned = (num_bytes + BUMP_ALLOC_ALIGN - 1) & ~(BUMP_ALLOC_ALIGN - 1);

    // Atomicly advance the offset
    // In single-thread POC we could skip atomics, but this is safer
    size_t old_offset = atomicAdd((unsigned long long *)&bump_heap_offset, (unsigned long long)aligned);

    if (old_offset + aligned > bump_heap_size) {
        // Out of memory - return NULL
        return (void *)0;
    }

    return (void *)(bump_heap_base + old_offset);
}

static __device__ inline void *bump_realloc(void *ptr, size_t new_num_bytes) {
    // Simple strategy: allocate new block and copy
    // We don't know old size, so we copy new_num_bytes (may read past old allocation,
    // but that's safe on GPU memory since the whole buffer is allocated)
    if (ptr == (void *)0) {
        return bump_malloc(new_num_bytes);
    }
    if (new_num_bytes == 0) {
        return (void *)0;
    }
    void *new_ptr = bump_malloc(new_num_bytes);
    if (new_ptr != (void *)0) {
        // Copy the data. We don't know the old size, so we just copy new_num_bytes.
        // Since both old and new are in the same contiguous buffer, reading past
        // the old allocation won't fault (just garbage data in the extra bytes,
        // which the caller will overwrite).
        char *dst = (char *)new_ptr;
        char *src = (char *)ptr;
        for (size_t i = 0; i < new_num_bytes; i++) {
            dst[i] = src[i];
        }
    }
    return new_ptr;
}

static __device__ inline void bump_free(void *ptr) {
    // Bump allocator: free is a no-op
    (void)ptr;
}

#endif // __CUDA_ARCH__

// Host-side initialization/cleanup (available in both CUDA and host builds)
#ifdef __cplusplus
extern "C" {
#endif
char *bump_alloc_init(size_t heap_size);
void bump_alloc_deinit(char *d_heap);
#ifdef __cplusplus
}
#endif

#endif // MICROPY_BUMPALLOC_H
