#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static inline void _gpuCheckImpl(cudaError_t code,
                                 const char *label,
                                 const char *file,
                                 int         line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",
                file, line, label, cudaGetErrorString(code));
        exit(1);
    }
}

// Public macro: wraps the expression and passes its text as the label.
#define catchError(expr) _gpuCheckImpl((expr), #expr, __FILE__, __LINE__)
