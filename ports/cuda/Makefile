# Makefile for CUDA port of MicroPython
# POC version - compiles with nvcc

# Include common MicroPython build environment
include ../../py/mkenv.mk

# CUDA compiler location
NVCC = /usr/local/cuda/bin/nvcc

# qstr definitions (must come before including py.mk)
QSTR_DEFS = qstrdefsport.h

# Disable ROM text compression for simplicity
MICROPY_ROM_TEXT_COMPRESSION = 0

# Include py core make definitions
include $(TOP)/py/py.mk

# Include directories
INC += -I.
INC += -I$(TOP)
INC += -I$(BUILD)

# For initial POC, compile with regular gcc to test
# Later we'll switch to nvcc for device code
CC = gcc
LD = gcc

# Compiler flags
CFLAGS += $(INC) -Wall -Werror -std=c99 -g
# MICROPY_NLR_SETJMP is set in mpconfigport.h

# Optimization
CFLAGS += -Os -DNDEBUG
CFLAGS += -fdata-sections -ffunction-sections

# Linker flags
LDFLAGS += -Wl,--gc-sections

# Port source files
SRC_C = \
	main.c \
	mphal.c \

# Object files
OBJ = $(PY_CORE_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))

# Build target
all: $(BUILD)/micropython

$(BUILD)/micropython: $(OBJ)
	$(ECHO) "LINK $@"
	$(Q)$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)
	$(Q)size $@

# Run target
run: $(BUILD)/micropython
	$(BUILD)/micropython

# Clean up generated headers on clean
clean: clean-genhdr

clean-genhdr:
	$(RM) -rf $(BUILD)/genhdr

# Include standard rules
include $(TOP)/py/mkrules.mk
