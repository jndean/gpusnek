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
 * Stack scanning strategy:
 * We read the hardware stack pointer directly via PTX inline asm (%SP).
 * This is the most accurate bottom-of-stack value.  MP_STATE_THREAD(stack_top)
 * is set at mp_init time (the highest stack address this thread will use),
 * giving us the range [%SP, stack_top) to scan conservatively for GC roots.
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

// Device implementation: read stack pointer via PTX %SP inline asm.
__device__ __noinline__ void gc_collect(void) {
    // Read the current stack pointer.  PTX %SP is the per-thread stack pointer
    // register, which always points to the current top of the call stack.
    void *sp;
    asm("mov.u64 %0, %%SP;" : "=l"(sp));

    char *stack_top = MP_STATE_THREAD(stack_top);
    if (stack_top == NULL) {
        // stack_top not yet set (called before mp_init completed)
        return;
    }

    // Delete these diagnostics later
    // On CUDA the stack grows downward: sp < stack_top.
    ptrdiff_t range = (char *)stack_top - (char *)sp;
    // Print diagnostic: this will appear in CUDA printf buffer even if kernel later crashes.
    printf("[GC] sp=%p top=%p range=%d\n", sp, stack_top, (int)range);
    if (range <= 0 || range > 512 * 1024) {
        // Sanity guard: don't scan a nonsensical range.
        printf("[GC] bad range %d, stack_top=%p sp=%p\n",
               (int)range, stack_top, sp);
        return;
    }

    gc_collect_start();
    gc_collect_root((void **)sp, (size_t)range / sizeof(void *));
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
