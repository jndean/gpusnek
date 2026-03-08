#include <stdio.h>
#include "py/mpconfig.h"
#include "py/obj.h"
#include "py/mperrno.h"
#include "py/lexer.h"
#include "py/compile.h"
#include "py/misc.h"
#include "py/builtin.h"


// Required stubs for MicroPython

// Lexer from file - not supported
MAYBE_CUDA mp_lexer_t *mp_lexer_new_from_file(qstr filename) {
    mp_raise_OSError(MP_ENOENT);
}

// Import stat - nothing exists
MAYBE_CUDA mp_import_stat_t mp_import_stat(const char *path) {
    return MP_IMPORT_STAT_NO_EXIST;
}

// NLR jump fail - called when an exception has no handler
// This is required by nlrsetjmp.c
MAYBE_CUDA void nlr_jump_fail(void *val) {
    printf("FATAL: Uncaught exception:\n");
    mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(val));
    #ifdef __CUDACC__
    asm("trap;");
    #endif
    while (1) { }
}

// Fatal error handler
MAYBE_CUDA void __fatal_error(const char *msg) {
    printf("FATAL ERROR: %s\n", msg);
    while (1) { }
}
