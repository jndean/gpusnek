#include <stdio.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"

// Execute a Python string
MAYBE_CUDA void do_str(const char *src, mp_parse_input_kind_t input_kind) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, strlen(src), 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t parse_tree = mp_parse(lex, input_kind);
        mp_obj_t module_fun = mp_compile(&parse_tree, source_name, true);
        mp_call_function_0(module_fun);
        nlr_pop();
    } else {
        printf("Exception occurred in do_str\n");
    }
}

MAYBE_CUDA void run_micropython_tests(void) {
    printf("Running MicroPython tests...\n");

    // Test 1: Basic arithmetic
    printf("Test 1: Basic arithmetic\n");
    do_str("print(1+2+3)", MP_PARSE_FILE_INPUT);

    // Test 2: Variables
    printf("Test 2: Variables\n");
    do_str("x = 42\nprint(x * 2)", MP_PARSE_FILE_INPUT);

    // Test 3: List comprehension
    printf("Test 3: List comprehension\n");
    do_str("squares = [x*x for x in range(5)]\nprint(squares)", MP_PARSE_FILE_INPUT);

    // Test 4: String formatting
    printf("Test 4: String formatting\n");
    do_str("name = 'CUDA'\nprint('Hello, {}!'.format(name))", MP_PARSE_FILE_INPUT);

    // Test 5: Class definition and method call
    printf("Test 5: Class definition\n");
    do_str(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "    def inc(self):\n"
        "        self.count += 1\n"
        "        return self.count\n"
        "c = Counter()\n"
        "print(c.inc(), c.inc(), c.inc())\n",
        MP_PARSE_FILE_INPUT);

    // Test 6: Monkey-patch a method
    printf("Test 6: Method patching\n");
    do_str(
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'Hello'\n"
        "def new_greet(self):\n"
        "    return 'Patched!'\n"
        "g = Greeter()\n"
        "Greeter.greet = new_greet\n"
        "print(g.greet())\n",
        MP_PARSE_FILE_INPUT);

    // Test 7: Lambda and higher-order functions
    printf("Test 7: Lambda and map\n");
    do_str("print(list(map(lambda x: x*2, [1,2,3])))", MP_PARSE_FILE_INPUT);

    // Test 8: Tuple unpacking
    printf("Test 8: Tuple unpacking\n");
    do_str("a, b, c = (10, 20, 30)\nprint(a + b + c)", MP_PARSE_FILE_INPUT);

    // Test 9: Dictionary
    printf("Test 9: Dictionary\n");
    do_str("d = {'a': 1, 'b': 2}\nprint(d['a'] + d['b'])", MP_PARSE_FILE_INPUT);

    // Test 10: Generator expression with sum
    printf("Test 10: Generator expression\n");
    do_str("print(sum(x for x in range(10)))", MP_PARSE_FILE_INPUT);

    // Test 11: Generator expression with sum
    printf("Test 11: Types\n");
    do_str("print(dir(type(type(1))))\n", MP_PARSE_FILE_INPUT);

    // Test 12: Per-thread __main__ module isolation
    // Each thread sets test_isolation to a DIFFERENT value (100 + thread_id),
    // then a separate do_str reads it back. If threads shared one __main__,
    // one thread would see the other's value.
    printf("Test 12: __main__ module isolation\n");
    {
        int tid = MP_THREAD_IDX;
        int val = 100 + tid;
        // Build "test_isolation = 1XX" — last digit varies by thread
        char set_src[] = "test_isolation = 100";
        set_src[19] = '0' + (val % 10);  // patch units digit
        set_src[18] = '0' + (val / 10) % 10;  // patch tens digit
        set_src[17] = '0' + (val / 100);  // patch hundreds digit
        do_str(set_src, MP_PARSE_FILE_INPUT);

        // Build "assert test_isolation == 1XX\nprint(test_isolation)"
        char chk_src[] = "assert test_isolation == 100\nprint(test_isolation)";
        chk_src[27] = '0' + (val % 10);
        chk_src[26] = '0' + (val / 10) % 10;
        chk_src[25] = '0' + (val / 100);
        do_str(chk_src, MP_PARSE_FILE_INPUT);
    }

    printf("MicroPython tests finished.\n");
}
