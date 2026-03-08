/*
 * Custom replacements for C standard library string functions.
 *
 * On CUDA, the standard string functions (strlen, strcmp, etc.) are not
 * available in device code. We provide our own implementations that work
 * on both host and device, so we can validate correctness on the host.
 *
 * Declarations are here; implementations are in cuda_string.c.
 */

#ifndef MICROPY_CUDA_STRING_H
#define MICROPY_CUDA_STRING_H

#include <stddef.h>

MAYBE_CUDA size_t __mp_strlen(const char *s);
MAYBE_CUDA int __mp_strcmp(const char *s1, const char *s2);
MAYBE_CUDA int __mp_strncmp(const char *s1, const char *s2, size_t n);
MAYBE_CUDA void *__mp_memcpy(void *dst, const void *src, size_t n);
MAYBE_CUDA void *__mp_memmove(void *dst, const void *src, size_t n);
MAYBE_CUDA int __mp_memcmp(const void *s1, const void *s2, size_t n);
MAYBE_CUDA void *__mp_memset(void *s, int c, size_t n);
MAYBE_CUDA char *__mp_strchr(const char *s, int c);
MAYBE_CUDA unsigned long __mp_strtoul(const char *nptr, char **endptr, int base);

// Redirect standard library calls to our implementations
#undef strlen
#undef strcmp
#undef strncmp
#undef memcpy
#undef memmove
#undef memcmp
#undef memset
#undef strchr
#undef strtoul
#define strlen(s) __mp_strlen(s)
#define strcmp(a, b) __mp_strcmp(a, b)
#define strncmp(a, b, n) __mp_strncmp(a, b, n)
#define memcpy(d, s, n) __mp_memcpy(d, s, n)
#define memmove(d, s, n) __mp_memmove(d, s, n)
#define memcmp(a, b, n) __mp_memcmp(a, b, n)
#define memset(d, c, n) __mp_memset(d, c, n)
#define strchr(s, c) __mp_strchr(s, c)
#define strtoul(s, e, b) __mp_strtoul(s, e, b)

#endif // MICROPY_CUDA_STRING_H
