#include <py/runtime.h>

MP_NORETURN MAYBE_CUDA void abort_(void);

MP_NORETURN MAYBE_CUDA void abort_(void) {
    mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("abort() called"));
}
