# Makefile for CUDA port of MicroPython
# Default target
TARGET ?= cuda

# Include common MicroPython build environment
include micropython/py/mkenv.mk

# CUDA compiler location
NVCC = /usr/local/cuda/bin/nvcc

# qstr definitions (must come before including py.mk)
QSTR_DEFS = qstrdefsport.h

# Disable ROM text compression for simplicity
MICROPY_ROM_TEXT_COMPRESSION = 0

# Define build directory based on target
BUILD = build-$(TARGET)

# Files in shared/ that contain qstrs/root-pointers and must be scanned
# before py.mk processes SRC_QSTR
SRC_QSTR += micropython/shared/runtime/pyexec.c \
            micropython/shared/readline/readline.c

# Include py core make definitions
include $(TOP)/py/py.mk

# Enable VFS and LFS2
MICROPY_VFS = 1
MICROPY_VFS_LFS2 = 1

# Include extmod make definitions
include $(TOP)/extmod/extmod.mk

# Include directories
INC += -I.
INC += -I$(TOP)
INC += -I$(BUILD)

CUDACFLAGS = $(INC) -x cu -dc -Xcompiler -fPIC -Xcompiler -fpermissive -D"restrict=" -G

ifeq ($(TARGET),cuda)
# CUDA Build Settings
CC = $(NVCC)
LD = $(NVCC)
# Optimization for nvcc (must be -O1, -O2, or -O3; -Os not supported)
COPT = -O1
CFLAGS = $(CUDACFLAGS)
CFLAGS += -O1 -DNDEBUG
# Overrides for nvcc (it doesn't support -Os)
$(PY_BUILD)/nlr%.o: CFLAGS = $(CUDACFLAGS) -O1 -DNDEBUG
else
# Host Build Settings (G++)
CC = g++
LD = g++
CFLAGS = $(INC) -Wall -Werror -x c++ -fPIC -fpermissive
CFLAGS += -O1 -DNDEBUG
endif

# Linker flags
LDFLAGS +=

# Port source files
SRC_C = \
	mphal.c \
	gcollect.c \
	gpusnek.c \
	port_stubs.c \
	micropython/shared/runtime/pyexec.c \
	micropython/shared/readline/readline.c

# Object files
OBJ = $(PY_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))

# Rule for .cu files
$(BUILD)/%.o: %.cu
	$(NVCC) $(CUDACFLAGS) -c $< -o $@

# Build target
LIB_OUT = $(BUILD)/libgpusnek.a
all: $(LIB_OUT)

$(LIB_OUT): $(OBJ)
ifeq ($(TARGET),cuda)
	$(ECHO) "NVCC-LIB $@"
	$(Q)$(NVCC) -lib -o $@ $^
else
	$(ECHO) "AR $@"
	$(Q)ar rcs $@ $^
endif
	$(Q)size $@



# Include standard rules
include $(TOP)/py/mkrules.mk
