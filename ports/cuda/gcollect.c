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

/*
 * gc_collect() has separate host and device implementations because
 * the device side uses PTX inline asm to read the hardware stack pointer.
 * We use __CUDA_ARCH__ (defined only during device compilation) to select
 * the right path, not __CUDACC__ (defined for both host and device passes).
 */

#ifdef __CUDA_ARCH__

// Device implementation: use both &dummy and PTX %SP, assert they agree
__device__ __noinline__ void gc_collect(void) {
    void *dummy;

    // Method 1: address of local variable (portable proxy for SP)
    void *sp_local = (void *)&dummy;

    // Method 2: read stack pointer via PTX inline asm
    void *sp_asm;
    asm("mov.u64 %0, %%SP;" : "=l"(sp_asm));

    // Verify both methods agree (sanity check)
    // They may differ by a small offset due to the local variable itself
    // but should be within one stack frame (~256 bytes)
    ptrdiff_t diff = (char *)sp_local - (char *)sp_asm;
    if (diff < 0) diff = -diff;
    if (diff > 256) {
        printf("FATAL: SP mismatch: local=%p asm=%p diff=%td\n",
               sp_local, sp_asm, diff);
        return;
    }

    // Use the lower of the two (scans more of the stack = safer)
    void *sp = (sp_local < sp_asm) ? sp_local : sp_asm;

    gc_collect_start();
    gc_collect_root((void **)sp,
                    ((mp_uint_t)MP_STATE_THREAD(stack_top) - (mp_uint_t)sp) / sizeof(void *));
    gc_collect_end();
}

#else

// Host implementation: use &dummy as SP proxy (same approach as ports/minimal)
__attribute__((noinline)) void gc_collect(void) {
    void *dummy;
    gc_collect_start();
    gc_collect_root(&dummy,
                    ((mp_uint_t)MP_STATE_THREAD(stack_top) - (mp_uint_t)&dummy) / sizeof(void *));
    gc_collect_end();
}

#endif // __CUDA_ARCH__

#endif // MICROPY_ENABLE_GC
