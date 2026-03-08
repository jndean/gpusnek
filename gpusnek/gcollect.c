/*
 * Garbage collection implementation for the CUDA port.
 *
 * This file provides gc_collect(), which is called by the GC when it needs
 * to find all root pointers on the C stack. It is intentionally in a
 * separate compilation unit from the functions that call allocating routines,
 * which helps prevent the compiler from inlining gc_collect and increases
 * the likelihood that callers must spill live registers to the stack before
 * calling into this file.
 *
 * KNOWN LIMITATION (callee-saved registers on CUDA):
 * On CPU ports, gc_collect() calls setjmp() which dumps all callee-saved
 * registers onto the stack before scanning.  CUDA has no setjmp, so we
 * rely on the calling convention: when a caller invokes gc_collect() (which
 * is __noinline__ and in a separate TU), the compiler must save any live
 * values it needs after the call to the stack.  However, values placed
 * in callee-saved registers that the callee does NOT itself use will NOT
 * be spilled — if one of these holds the sole pointer to a live GC object,
 * the GC may incorrectly free it.  In practice, MicroPython's VM stores
 * most state through memory indirection (MP_STATE_VM/MP_STATE_CTX) and the
 * deep call chain to gc_collect forces extensive spilling, so the risk is
 * low.  If issues arise, a possible fix is implementing a PTX-level
 * register dump or switching to a precise rooting scheme.
 */

#include "py/mpconfig.h"

#if MICROPY_ENABLE_GC

#include <stdio.h>
#include "py/gc.h"
#include "py/mpstate.h"


MAYBE_CUDA __attribute__((noinline)) void gc_collect(void) {
    void *sp;
    // We try really hard to convince nvcc to generate a stack frame for this
    // function, otherwise the top stack pointer will will be garbage.
    volatile int dummy = 0;
    sp = (void*)&dummy;
    gc_collect_start();
    gc_collect_root((void **)sp,
        ((mp_uint_t)MP_STATE_THREAD(stack_top) - (mp_uint_t)sp) / sizeof(void *));
    gc_collect_end();
}

#endif // MICROPY_ENABLE_GC
