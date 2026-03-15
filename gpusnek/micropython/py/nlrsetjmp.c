/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2013-2023 Damien P. George
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "py/mpstate.h"
#include "py/obj.h"
#include "py/runtime.h"
#if MICROPY_ENABLE_GC
#include "py/gc.h"
#endif

#if MICROPY_HELPER_REPL
#include "shared/readline/readline.h"
#include "py/mphal.h"
#endif

#if MICROPY_NLR_SETJMP

MAYBE_CUDA void nlr_jump(void *val) {
#ifdef __CUDA_ARCH__
    // Device code cannot use longjmp.
    // Guard against recursive entry (e.g., if mp_obj_print_exception itself OOMs).
    static __device__ bool in_nlr_jump = false;
    if (in_nlr_jump) {
        printf("FATAL: recursive exception in nlr_jump\n");
        // Still reset gc_lock_depth so allocations can continue.
        #if MICROPY_ENABLE_GC
        MP_STATE_THREAD(gc_lock_depth) = 0;
        #endif
        asm volatile ("exit;");
    }
    in_nlr_jump = true;
    mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(val));
    in_nlr_jump = false;
    // Release any GC locks held when the exception was thrown.
    // Without this, an OOM during compilation (when gc_lock_depth > 0) would
    // permanently prevent future allocations.
    #if MICROPY_ENABLE_GC
    MP_STATE_THREAD(gc_lock_depth) = 0;
    #endif

    #if MICROPY_HELPER_REPL
    if (MP_STATE_VM(repl_line)) {
        MP_STATE_CTX.repl_state.repl.cont_line = false;
        MP_STATE_CTX.repl_state.repl.paste_mode = false;
        mp_hal_stdout_tx_str("\r\n");
        readline_init(MP_STATE_VM(repl_line), ">>> ");
    }
    #endif

    // Abort thread instead of returning to prevent subsequent illegal memory accesses
    asm volatile ("exit;");
#else
    MP_NLR_JUMP_HEAD(val, top);
    longjmp(top->jmpbuf, 1);
#endif
}

#endif
