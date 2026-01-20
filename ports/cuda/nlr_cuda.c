// NLR (Non-Local Return) implementation for CUDA
// Since we can't use setjmp/longjmp properly in CUDA, we implement
// a version that terminates the thread on exception (POC approach)
//
// When MICROPY_NLR_SETJMP is enabled, nlr_push is a macro that calls
// nlr_push_tail() then setjmp(). We provide nlr_push_tail and nlr_jump.

#include "py/mpstate.h"
#include "py/nlr.h"

#include <stdio.h>
#include <setjmp.h>

#if MICROPY_NLR_SETJMP

// nlr_push_tail - called by the nlr_push macro after the buffer is set up
// We just add the buffer to the linked list
unsigned int nlr_push_tail(nlr_buf_t *top) {
    top->prev = MP_STATE_THREAD(nlr_top);
    MP_STATE_THREAD(nlr_top) = top;
    return 0;  // Return value not used by caller
}

// nlr_pop - remove the top NLR buffer from the chain
void nlr_pop(void) {
    MP_STATE_THREAD(nlr_top) = MP_STATE_THREAD(nlr_top)->prev;
}

// nlr_jump - called when an exception is raised
// In the standard setjmp version, this does longjmp.
// For CUDA POC, we can't really do a non-local jump, so we try longjmp
// and if that fails (or for CUDA device code), we call the fail handler.
void nlr_jump(void *val) {
    nlr_buf_t **top_ptr = &MP_STATE_THREAD(nlr_top);
    nlr_buf_t *top = *top_ptr;
    
    if (top == NULL) {
        // No exception handler, call fail
        nlr_jump_fail(val);
        // nlr_jump_fail doesn't return, but just in case:
        for (;;) { }
    }
    
    // Store the exception value
    top->ret_val = val;
    
    // Restore the previous handler
    *top_ptr = top->prev;
    
    // Try to do the longjmp
    // This works for host code, for CUDA device code it will fail
    longjmp(top->jmpbuf, 1);
}

// nlr_jump_fail - called when there's no exception handler
// This terminates the CUDA thread
void nlr_jump_fail(void *val) {
    printf("CUDA MicroPython: Unhandled exception, terminating\n");
    
#ifdef __CUDA_ARCH__
    // In CUDA device code, use PTX exit to terminate this thread
    asm("exit;");
#else
    // For host code, infinite loop (should not normally be reached)
    while (1) { }
#endif
}

#endif // MICROPY_NLR_SETJMP
