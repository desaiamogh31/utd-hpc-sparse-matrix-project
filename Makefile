# =============================================================================
# Top-level Makefile — UTD HPC Sparse Matrix Project
# =============================================================================

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS  =

# Optional: enable OpenMP
ifdef OMP
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

# Optional: enable MPI (switches compiler to mpicc)
ifdef MPI
  CC = mpicc
endif

# Directories
SRC_DIR   = src
BIN_DIR   = bin
OBJ_DIR   = obj

# Library source files (no main() — subdirectories only)
LIB_SRCS := $(wildcard $(SRC_DIR)/formats/*.c \
                        $(SRC_DIR)/assembly/*.c \
                        $(SRC_DIR)/kernels/*.c \
                        $(SRC_DIR)/utils/*.c)
LIB_OBJS := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(LIB_SRCS))

.PHONY: all clean tests bench dirs

all: dirs $(BIN_DIR)/spmv $(BIN_DIR)/assembly

dirs:
	@mkdir -p $(BIN_DIR) $(OBJ_DIR) \
	           $(OBJ_DIR)/formats \
	           $(OBJ_DIR)/assembly \
	           $(OBJ_DIR)/kernels \
	           $(OBJ_DIR)/utils

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Each executable has its own main source file in src/
$(BIN_DIR)/spmv: $(OBJ_DIR)/spmv_main.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/assembly: $(OBJ_DIR)/assembly_main.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/spmv_main.o: $(SRC_DIR)/spmv_main.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/assembly_main.o: $(SRC_DIR)/assembly_main.c
	$(CC) $(CFLAGS) -c $< -o $@

tests: dirs $(LIB_OBJS)
	@echo "Building and running tests..."
	@for f in tests/*.c; do \
	    [ -f "$$f" ] || continue; \
	    name=$$(basename $$f .c); \
	    $(CC) $(CFLAGS) $$f $(LIB_OBJS) -o $(BIN_DIR)/$$name $(LDFLAGS) && \
	    $(BIN_DIR)/$$name && echo "PASS: $$name" || echo "FAIL: $$name"; \
	done

bench: all
	@echo "Running benchmarks..."
	@mkdir -p results
	@for f in benchmarks/*.sh; do \
	    [ -f "$$f" ] || continue; \
	    bash $$f | tee results/$$(basename $$f .sh).txt; \
	done

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) results
