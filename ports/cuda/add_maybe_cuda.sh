#!/bin/bash
# Conservative bulk MAYBE_CUDA annotation script
# Adds MAYBE_CUDA to function declarations/definitions in py/*.h and py/*.c
# 
# SAFETY: Only modifies patterns at the start of a line (column 0)
#         Skips lines already containing MAYBE_CUDA
#         Does NOT handle multiline declarations
#
# USAGE: 
#   ./add_maybe_cuda.sh --dry-run   # Preview changes
#   ./add_maybe_cuda.sh             # Apply changes
#
# VERIFY: After running, test with:
#   make TARGET=host clean && make TARGET=host && ./build-host/micropython -c "print('OK')"

set -e

PY_DIR="../../py"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be modified ==="
fi

# Function to process a file with sed patterns
# Args: $1 = file, $2 = sed pattern, $3 = description
apply_sed() {
    local file="$1"
    local pattern="$2"
    local desc="$3"
    
    if $DRY_RUN; then
        # Show what would change
        local matches=$(grep -n "$pattern" "$file" 2>/dev/null | grep -v "MAYBE_CUDA" | head -5)
        if [[ -n "$matches" ]]; then
            echo "[$desc] $file:"
            echo "$matches"
            echo ""
        fi
    else
        # Apply the change, but only to lines NOT already containing MAYBE_CUDA
        sed -i "/$pattern/{/MAYBE_CUDA/!s/$pattern/MAYBE_CUDA &/}" "$file"
    fi
}

echo "Processing header files (.h)..."
echo "================================"

for f in "$PY_DIR"/*.h; do
    # Skip if file doesn't exist (glob didn't match)
    [[ -f "$f" ]] || continue
    
    # Function declarations: return_type function_name(...)
    # Pattern: Start of line, return type, space, mp_* or m_* function name
    
    # void mp_* functions
    apply_sed "$f" '^void mp_[a-z_]*(' "void mp_*()"
    
    # bool mp_* functions  
    apply_sed "$f" '^bool mp_[a-z_]*(' "bool mp_*()"
    
    # mp_obj_t mp_* functions
    apply_sed "$f" '^mp_obj_t mp_[a-z_]*(' "mp_obj_t mp_*()"
    
    # mp_int_t mp_* functions
    apply_sed "$f" '^mp_int_t mp_[a-z_]*(' "mp_int_t mp_*()"
    
    # mp_uint_t mp_* functions
    apply_sed "$f" '^mp_uint_t mp_[a-z_]*(' "mp_uint_t mp_*()"
    
    # size_t mp_* functions
    apply_sed "$f" '^size_t mp_[a-z_]*(' "size_t mp_*()"
    
    # const char * functions
    apply_sed "$f" '^const char \*mp_[a-z_]*(' "const char *mp_*()"
    apply_sed "$f" '^const char \*qstr_[a-z_]*(' "const char *qstr_*()"
    
    # const byte * functions
    apply_sed "$f" '^const byte \*qstr_[a-z_]*(' "const byte *qstr_*()"
    
    # mp_float_t functions
    apply_sed "$f" '^mp_float_t mp_[a-z_]*(' "mp_float_t mp_*()"
    
    # long long functions
    apply_sed "$f" '^long long mp_[a-z_]*(' "long long mp_*()"
    
    # Memory functions (m_*)
    apply_sed "$f" '^void \*m_[a-z_]*(' "void *m_*()"
    apply_sed "$f" '^void m_[a-z_]*(' "void m_*()"
    
    # extern declarations for types/objects
    apply_sed "$f" '^extern const mp_obj_type_t ' "extern const mp_obj_type_t"
    apply_sed "$f" '^extern const mp_obj_t ' "extern const mp_obj_t"
    apply_sed "$f" '^extern const mp_print_t ' "extern const mp_print_t"
    
    # MP_NORETURN functions (raise functions)
    apply_sed "$f" '^MP_NORETURN void mp_raise' "MP_NORETURN void mp_raise*"
done

echo ""
echo "Processing source files (.c)..."
echo "================================"

for f in "$PY_DIR"/*.c; do
    [[ -f "$f" ]] || continue
    
    # Function definitions (similar patterns but in .c files)
    apply_sed "$f" '^void mp_[a-z_]*(' "void mp_*()"
    apply_sed "$f" '^bool mp_[a-z_]*(' "bool mp_*()"
    apply_sed "$f" '^mp_obj_t mp_[a-z_]*(' "mp_obj_t mp_*()"
    apply_sed "$f" '^mp_int_t mp_[a-z_]*(' "mp_int_t mp_*()"
    apply_sed "$f" '^mp_uint_t mp_[a-z_]*(' "mp_uint_t mp_*()"
    apply_sed "$f" '^size_t mp_[a-z_]*(' "size_t mp_*()"
    apply_sed "$f" '^const char \*mp_[a-z_]*(' "const char *mp_*()"
    apply_sed "$f" '^mp_float_t mp_[a-z_]*(' "mp_float_t mp_*()"
    apply_sed "$f" '^long long mp_[a-z_]*(' "long long mp_*()"
    
    # Memory functions
    apply_sed "$f" '^void \*m_[a-z_]*(' "void *m_*()"
    apply_sed "$f" '^void m_[a-z_]*(' "void m_*()"
    
    # qstr functions
    apply_sed "$f" '^const char \*qstr_[a-z_]*(' "const char *qstr_*()"
    apply_sed "$f" '^const byte \*qstr_[a-z_]*(' "const byte *qstr_*()"
    apply_sed "$f" '^qstr qstr_[a-z_]*(' "qstr qstr_*()"
    apply_sed "$f" '^size_t qstr_[a-z_]*(' "size_t qstr_*()"
    
    # Print functions
    apply_sed "$f" '^int mp_printf(' "int mp_printf()"
    apply_sed "$f" '^int mp_vprintf(' "int mp_vprintf()"
    apply_sed "$f" '^int mp_print_[a-z_]*(' "int mp_print_*()"
    
    # MP_NORETURN functions
    apply_sed "$f" '^MP_NORETURN void mp_raise' "MP_NORETURN void mp_raise*"
done

if $DRY_RUN; then
    echo ""
    echo "=== DRY RUN COMPLETE ==="
    echo "Run without --dry-run to apply changes"
else
    echo ""
    echo "=== CHANGES APPLIED ==="
    echo ""
    echo "Next steps:"
    echo "1. Test host build:  make TARGET=host clean && make TARGET=host"
    echo "2. Run test:         ./build-host/micropython -c \"print('OK')\""
    echo "3. Test CUDA build:  make TARGET=cuda build-cuda/py/obj.o"
fi
