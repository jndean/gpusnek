#include <stdio.h>
#include <string.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/mperrno.h"

#include "tests.h"



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

    printf("Test 13: GC\n");
    do_str(
        "x = 1\n"
        "z = []\n"
        "for y in range(10000):\n"
        "    x += 1\n"
        "    y = [x, x+1, x+2]\n"
        "    z.append(y)\n"
        "    if len(z) > 6:\n"
        "        z.pop(0)\n"
        "print(z[-1])\n"
        , MP_PARSE_FILE_INPUT
    );

    // printf("Test 14: Exception\n");
    // do_str(
    //     "x = [1,2,3]\n"
    //     "print(x[10])\n",
    //     MP_PARSE_FILE_INPUT
    // );

    // Test 15: mp_bind_array — write to a C buffer from Python
    printf("Test 15: mp_bind_array\n");
    static unsigned char shared_buf[8] = {0};
    mp_bind_array("data", shared_buf, 8);
    do_str(
        "data[0] = 42\n"
        "data[7] = 255\n"
        "print(len(data), data[0], data[7])\n",
        MP_PARSE_FILE_INPUT
    );
    // Verify the writes actually landed in the C buffer
    printf("[C] shared_buf[0]=%d shared_buf[7]=%d\n",
           (int)shared_buf[0], (int)shared_buf[7]);

    // Test 16: syncthreads keyword
    printf("Test 16: syncthreads keyword\n");
    do_str(
        "syncthreads\n",
        MP_PARSE_FILE_INPUT
    );

    printf("MicroPython tests finished.\n");
}
