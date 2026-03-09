#include <stdio.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"
#include "py/objarray.h"
#include "py/obj.h"
#include "py/pystack.h"
#include "py/bumpalloc.h"

#include "gpusnek.h"


// Initialize Port environment (allocators, state pointers, per-thread __main__)
// Needs to be called before running MicroPython.
MAYBE_CUDA void gpusnek_init(mp_state_ctx_t *ctx, void *stack, size_t stack_size, void *heap, size_t heap_size) {
    // Set up per-thread state and allocator
    mp_state_ctx_array = ctx;
    memset(&MP_STATE_CTX, 0, sizeof(mp_state_ctx_t));
    
    #if MICROPY_ENABLE_PYSTACK
    mp_pystack_init(stack, (uint8_t *)stack + stack_size);
    #endif

    #if MICROPY_ENABLE_GC
    gc_init(heap, (char *)heap + heap_size);
    #else
    bump_alloc_init(heap, heap_size);
    #endif

    #ifdef __CUDA_ARCH__
    // Break circular dependency for mp_type_type on device
    ((mp_obj_type_t *)&mp_type_type)->base.type = &mp_type_type;
    // Break circular dependency for dict_locals_dict on device
    extern MAYBE_CUDA mp_obj_dict_t dict_locals_dict;
    dict_locals_dict.base.type = &mp_type_dict;
    #endif

    // Initialize this thread's __main__ module
    mp_module___main__.base.type = &mp_type_module;
    mp_module___main__.globals = (mp_obj_dict_t *)&MP_STATE_VM(dict_main);

    mp_init();
}

MAYBE_CUDA void gpusnek_deinit(void) {
    mp_deinit();
}

// Execute a Python string
MAYBE_CUDA void gpusnek_do_str(const char *src, mp_parse_input_kind_t input_kind) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, strlen(src), 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t parse_tree = mp_parse(lex, input_kind);
        mp_obj_t module_fun = mp_compile(&parse_tree, source_name, false);
        mp_call_function_0(module_fun);
        nlr_pop();
    } else {
        printf("Exception occurred in gpusnek_do_str\n");
        // An exception (like MemoryError) during compilation might have jumped out
        // while the GC was locked. Reset lock depth so future allocations work.
        MP_STATE_THREAD(gc_lock_depth) = 0;
    }
}

// Bind an externally-owned array into the Python __main__ namespace.
// Creates a writable memoryview over [start, start+len) and stores it as
// a global named `name`.  The caller owns the memory; MicroPython never
// frees or resizes it.  Must be called after mp_init().
MAYBE_CUDA void gpusnek_bind_memory(const char *name, void *start, int len, char typecode) {
    // Build a writable memoryview over the external buffer with the specified typecode.
    mp_obj_t mv = mp_obj_new_memoryview(
        typecode | MP_OBJ_ARRAY_TYPECODE_FLAG_RW,
        (size_t)len,
        start
    );

    // Store it in __main__'s globals dict under the given name.
    mp_obj_t key = mp_obj_new_str(name, strlen(name));
    mp_obj_dict_store(mp_module___main__.globals, key, mv);
}

// Binds a C integer to the Python __main__ namespace.
MAYBE_CUDA void gpusnek_new_int(const char *name, int val) {
    mp_obj_t int_obj = mp_obj_new_int(val);
    mp_obj_t key = mp_obj_new_str(name, strlen(name));
    mp_obj_dict_store(mp_module___main__.globals, key, int_obj);
}

