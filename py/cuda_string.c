/*
 * Custom replacements for C standard library string functions.
 *
 * These implementations work on both host and CUDA device, allowing
 * validation on the host before running on the GPU.
 */

#include "py/mpconfig.h"

MAYBE_CUDA size_t __mp_strlen(const char *s) {
    size_t len = 0;
    while (s[len] != '\0') {
        len++;
    }
    return len;
}

MAYBE_CUDA int __mp_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

MAYBE_CUDA int __mp_strncmp(const char *s1, const char *s2, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s1[i] != s2[i]) {
            return (unsigned char)s1[i] - (unsigned char)s2[i];
        }
        if (s1[i] == '\0') {
            return 0;
        }
    }
    return 0;
}

MAYBE_CUDA void *__mp_memcpy(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
    return dst;
}

MAYBE_CUDA void *__mp_memmove(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    if (d < s) {
        for (size_t i = 0; i < n; i++) {
            d[i] = s[i];
        }
    } else if (d > s) {
        for (size_t i = n; i > 0; i--) {
            d[i - 1] = s[i - 1];
        }
    }
    return dst;
}

MAYBE_CUDA int __mp_memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;
    for (size_t i = 0; i < n; i++) {
        if (p1[i] != p2[i]) {
            return p1[i] - p2[i];
        }
    }
    return 0;
}

MAYBE_CUDA void *__mp_memset(void *s, int c, size_t n) {
    unsigned char *p = (unsigned char *)s;
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char)c;
    }
    return s;
}

MAYBE_CUDA char *__mp_strchr(const char *s, int c) {
    while (*s != '\0') {
        if (*s == (char)c) {
            return (char *)s;
        }
        s++;
    }
    if (c == '\0') {
        return (char *)s;
    }
    return (char *)0;
}

MAYBE_CUDA unsigned long __mp_strtoul(const char *nptr, char **endptr, int base) {
    unsigned long result = 0;
    const char *s = nptr;
    // Skip whitespace
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    // Handle base prefix
    if (base == 0 || base == 16) {
        if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            if (base == 0) base = 16;
            s += 2;
        } else if (base == 0 && s[0] == '0') {
            base = 8;
            s++;
        } else if (base == 0) {
            base = 10;
        }
    }
    while (*s) {
        int digit;
        if (*s >= '0' && *s <= '9') digit = *s - '0';
        else if (*s >= 'a' && *s <= 'f') digit = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') digit = *s - 'A' + 10;
        else break;
        if (digit >= base) break;
        result = result * base + digit;
        s++;
    }
    if (endptr) *endptr = (char *)s;
    return result;
}
