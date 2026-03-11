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

# Find all example directories
EXAMPLES = $(wildcard example_*)

.PHONY: all clean $(EXAMPLES)

ifndef EXAMPLE

# --- Top-Level Build ---
all: $(EXAMPLES)

# Always ensure the library is built first
.PHONY: FORCE_LIB
$(LIB_FILE): FORCE_LIB
	$(MAKE) -C $(LIB_DIR) TARGET=$(TARGET) -j

$(EXAMPLES): $(LIB_FILE)
	@echo "Building $@"
	$(MAKE) EXAMPLE=$@ TARGET=$(TARGET)

clean:
	@for dir in $(EXAMPLES); do \
		rm -f $$dir/*.o $$dir/$$(basename $$dir); \
	done
	$(MAKE) -C $(LIB_DIR) TARGET=cuda clean
	$(MAKE) -C $(LIB_DIR) TARGET=host clean

else

# --- Sub-Make Build for a Specific Example ---
SRC_C = $(wildcard $(EXAMPLE)/*.c)
SRC_CU = $(wildcard $(EXAMPLE)/*.cu)
OBJ = $(SRC_C:.c=-$(TARGET).o)
OBJ += $(SRC_CU:.cu=-$(TARGET).o)

$(EXAMPLE)/$(EXAMPLE): $(OBJ) $(LIB_FILE)
	$(CC) -o $@ $(OBJ) $(LIB_FILE) $(LDFLAGS)

%-$(TARGET).o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%-$(TARGET).o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

endif
