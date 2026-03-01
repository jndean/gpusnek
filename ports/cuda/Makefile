# Makefile for CUDA port of MicroPython
# POC version - compiles with nvcc

# Default target
TARGET ?= cuda

# Include common MicroPython build environment
include ../../py/mkenv.mk

# CUDA compiler location
NVCC = /usr/local/cuda/bin/nvcc

# qstr definitions (must come before including py.mk)
QSTR_DEFS = qstrdefsport.h

# Disable ROM text compression for simplicity
MICROPY_ROM_TEXT_COMPRESSION = 0

# Define build directory based on target
BUILD = build-$(TARGET)

# Include py core make definitions
include $(TOP)/py/py.mk

# Include directories
INC += -I.
INC += -I$(TOP)
INC += -I$(BUILD)

CUDACFLAGS = $(INC) -x cu -dc -Xcompiler -fpermissive -D"restrict="

ifeq ($(TARGET),cuda)
# CUDA Build Settings
CC = $(NVCC)
LD = $(NVCC)
# Optimization for nvcc (must be -O1, -O2, or -O3; -Os not supported)
COPT = -O1
CFLAGS = $(CUDACFLAGS)
CFLAGS += -O1 -DNDEBUG
# Overrides for nvcc (it doesn't support -Os)
$(PY_BUILD)/nlr%.o: CFLAGS = $(INC) -x cu -dc -Xcompiler -fpermissive -D"restrict=" -O1 -DNDEBUG
else
# Host Build Settings (G++)
CC = g++
LD = g++
CFLAGS = $(INC) -Wall -Werror -x c++ -fpermissive
CFLAGS += -O1 -DNDEBUG
endif

# Linker flags
LDFLAGS +=

# Port source files
SRC_C = \
	main.c \
	mphal.c \

# Object files
OBJ = $(PY_CORE_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))
OBJ += $(BUILD)/tests.o

ifeq ($(TARGET),cuda)
OBJ += $(BUILD)/cuda_kernel.o
endif

# Rule for .cu files
$(BUILD)/%.o: %.cu
	$(NVCC) $(CUDACFLAGS) -c $< -o $@

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
	$(RM) -rf build-cuda build-host

clean-genhdr:
	$(RM) -rf build-cuda/genhdr build-host/genhdr

# Include standard rules
include $(TOP)/py/mkrules.mk
