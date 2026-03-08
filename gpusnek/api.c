#include <stdio.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"
#include "py/objarray.h"
#include "py/obj.h"

#include "gpusnek_api.h"


// Execute a Python string
MAYBE_CUDA void do_str(const char *src, mp_parse_input_kind_t input_kind) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, strlen(src), 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t parse_tree = mp_parse(lex, input_kind);
        mp_obj_t module_fun = mp_compile(&parse_tree, source_name, false);
        mp_call_function_0(module_fun);
        nlr_pop();
    } else {
        printf("Exception occurred in do_str\n");
        // An exception (like MemoryError) during compilation might have jumped out
        // while the GC was locked. Reset lock depth so future allocations work.
        MP_STATE_THREAD(gc_lock_depth) = 0;
    }
}

// Bind an externally-owned byte array into the Python __main__ namespace.
// Creates a writable memoryview over [start, start+len) and stores it as
// a global named `name`.  The caller owns the memory; MicroPython never
// frees or resizes it.  Must be called after mp_init().
MAYBE_CUDA void mp_bind_array(const char *name, void *start, int len) {
    // Build a writable memoryview (typecode 'B' = uint8) over the external buffer.
    mp_obj_t mv = mp_obj_new_memoryview(
        'B' | MP_OBJ_ARRAY_TYPECODE_FLAG_RW,
        (size_t)len,
        start
    );

    // Store it in __main__'s globals dict under the given name.
    mp_obj_t key = mp_obj_new_str(name, strlen(name));
    mp_obj_dict_store(mp_module___main__.globals, key, mv);
}

