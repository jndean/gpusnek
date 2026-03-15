# Makefile for gpusnek root examples

TARGET ?= cuda

# Subdirectory containing the library
LIB_DIR = gpusnek

# Library build path based on target
LIB_BUILD_DIR = $(LIB_DIR)/build-$(TARGET)
LIB_FILE = $(LIB_BUILD_DIR)/libgpusnek.a

# Include directories
INC = -I. -I$(LIB_DIR) -I$(LIB_DIR)/micropython -I$(LIB_BUILD_DIR)

# Build Settings
ifeq ($(TARGET),cuda)
CC = /usr/local/cuda/bin/nvcc
CFLAGS = $(INC) -x cu -dc -Xcompiler -fpermissive -D"restrict=" -O1 -DNDEBUG -rdc=true
LDFLAGS = -lcudadevrt
else
CC = g++
CFLAGS = $(INC) -Wall -Werror -x c++ -fpermissive -O1 -DNDEBUG
LDFLAGS =
endif

# Find all example source files in root directory
SRC_C = $(wildcard example_*.c)
SRC_CU = $(wildcard example_*.cu)

# Generate names for the output executables
EXECUTABLES = $(SRC_C:.c=) $(SRC_CU:.cu=)

.PHONY: all clean $(EXECUTABLES)

all: $(EXECUTABLES)

# Always ensure the library is built first
.PHONY: FORCE_LIB
$(LIB_FILE): FORCE_LIB
	$(MAKE) -C $(LIB_DIR) TARGET=$(TARGET) -j

$(EXECUTABLES): %: %-$(TARGET).o $(LIB_FILE)
	@echo "Linking $@"
	$(CC) -o $@ $< $(LIB_FILE) $(LDFLAGS)

%-$(TARGET).o: %.c $(LIB_FILE)
	$(CC) $(CFLAGS) -c $< -o $@

%-$(TARGET).o: %.cu $(LIB_FILE)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f example_*-*.o
	rm -f $(EXECUTABLES)
	$(MAKE) -C $(LIB_DIR) TARGET=cuda clean
	$(MAKE) -C $(LIB_DIR) TARGET=host clean
