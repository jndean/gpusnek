// HAL (Hardware Abstraction Layer) implementation for CUDA/host
// Provides I/O backed by per-thread configurable buffers (see gpusnek_io_t),
// with printf as fallback when no buffer has been set.

#include <stdio.h>
#include <string.h>

#include "py/mpconfig.h"
#include "py/mphal.h"
#include "py/mpstate.h"

// Read a character from stdin.
// Uses the per-thread stdin buffer if configured; otherwise returns -1.
MAYBE_CUDA int mp_hal_stdin_rx_chr(void) {
    gpusnek_io_t *io = &MP_STATE_CTX.io;
    if (io->stdin_buf != NULL) {
        if (io->stdin_pos >= io->stdin_size) {
            return -1;  // past end of provided buffer
        }
        return (unsigned char)io->stdin_buf[io->stdin_pos++];
    }
    return -1;  // no stdin buffer configured
}


MAYBE_CUDA void append_stdout_overflow_msg() {
    gpusnek_io_t *io = &MP_STATE_CTX.io;
    const char msg[17] = "STDOUT OVERFLOW\n";
    char* write_head = io->stdout_buf + io->stdout_size - 17;
    for (int i = 0; i < 17; ++i) {
        write_head[i] = msg[i];
    }
}

// Write a string to stdout.
// Uses the per-thread stdout buffer if configured; truncates at buffer end.
// Falls back to printf when no buffer is set.
MAYBE_CUDA mp_uint_t mp_hal_stdout_tx_strn(const char *str, size_t len) {
    gpusnek_io_t *io = &MP_STATE_CTX.io;
    if (io->stdout_buf != NULL) {
        int remaining = io->stdout_size - io->stdout_pos - 1; // reserve 1 for NULL
        if (remaining <= 0) {
            append_stdout_overflow_msg();
            return 0;  // buffer full
        }
        size_t to_write = len < (size_t)remaining ? len : (size_t)remaining;
        memcpy(io->stdout_buf + io->stdout_pos, str, to_write);
        io->stdout_pos += (int)to_write;
        io->stdout_buf[io->stdout_pos] = '\0';  // keep NUL-terminated

        if (len > to_write) append_stdout_overflow_msg();
        return (mp_uint_t)to_write;
    }
    // fallback: printf
    for (size_t i = 0; i < len; i++) {
        printf("%c", str[i]);
    }
    return (mp_uint_t)len;
}

// Write a null-terminated string to stdout
MAYBE_CUDA void mp_hal_stdout_tx_str(const char *str) {
    mp_hal_stdout_tx_strn(str, strlen(str));
}

// Cooked variant (same as raw for our port)
MAYBE_CUDA void mp_hal_stdout_tx_strn_cooked(const char *str, size_t len) {
    mp_hal_stdout_tx_strn(str, len);
}

// Timing functions - return 0 (no timing support, TODO)
MAYBE_CUDA mp_uint_t mp_hal_ticks_ms(void) {
    return 0;
}

MAYBE_CUDA mp_uint_t mp_hal_ticks_us(void) {
    return 0;
}

MAYBE_CUDA mp_uint_t mp_hal_ticks_cpu(void) {
    return 0;
}

// Delay functions - no-op (TODO)
MAYBE_CUDA void mp_hal_delay_ms(mp_uint_t ms) {
    (void)ms;
}

MAYBE_CUDA void mp_hal_delay_us(mp_uint_t us) {
    (void)us;
}

// Set interrupt character - not supported
MAYBE_CUDA void mp_hal_set_interrupt_char(int c) {
    (void)c;
}
