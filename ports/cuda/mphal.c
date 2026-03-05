// HAL (Hardware Abstraction Layer) implementation for CUDA
// Provides I/O using printf and a fixed input buffer

#include <stdio.h>
#include <string.h>

#include "py/mpconfig.h"
#include "py/mphal.h"

// Fixed Python program to execute (compile-time input)
// This can be changed to run different programs
MAYBE_CUDA static const char *stdin_buffer = "print(1+2+3)\n";
MAYBE_CUDA static int stdin_pos = 0;

// Read a character from stdin (our fixed buffer)
MAYBE_CUDA int mp_hal_stdin_rx_chr(void) {
    char c = stdin_buffer[stdin_pos];
    if (c == '\0') {
        return -1;  // EOF
    }
    stdin_pos++;
    return (int)c;
}

// Write a string to stdout using printf
MAYBE_CUDA mp_uint_t mp_hal_stdout_tx_strn(const char *str, size_t len) {
    // Print character by character (CUDA printf has buffer limits)
    for (size_t i = 0; i < len; i++) {
        printf("%c", str[i]);
    }
    return len;
}

// Write a null-terminated string to stdout
MAYBE_CUDA void mp_hal_stdout_tx_str(const char *str) {
    printf("%s", str);
}

// Write a single character
MAYBE_CUDA void mp_hal_stdout_tx_strn_cooked(const char *str, size_t len) {
    mp_hal_stdout_tx_strn(str, len);
}

// Timing functions - return 0 for POC (no timing support)
MAYBE_CUDA mp_uint_t mp_hal_ticks_ms(void) {
    return 0;
}

MAYBE_CUDA mp_uint_t mp_hal_ticks_us(void) {
    return 0;
}

MAYBE_CUDA mp_uint_t mp_hal_ticks_cpu(void) {
    return 0;
}

// Delay functions - no-op for POC
MAYBE_CUDA void mp_hal_delay_ms(mp_uint_t ms) {
    (void)ms;
}

MAYBE_CUDA void mp_hal_delay_us(mp_uint_t us) {
    (void)us;
}

// Set interrupt character - no-op for CUDA
MAYBE_CUDA void mp_hal_set_interrupt_char(int c) {
    (void)c;
}
