# =============================================================================
# Top-level Makefile — UTD HPC Sparse Matrix Project
# =============================================================================

CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -Iinclude
LDFLAGS  =

# Optional: enable OpenMP
ifdef OMP
  CFLAGS  += -fopenmp
endif

# Optional: enable MPI (switches compiler to mpicc)
ifdef MPI
  CC = mpicc
endif

# Directories
SRC_DIR   = src
BIN_DIR   = bin
OBJ_DIR   = obj

SRCS   := $(wildcard $(SRC_DIR)/**/*.c $(SRC_DIR)/*.c)
OBJS   := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

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

$(BIN_DIR)/spmv: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/assembly: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests: dirs
	@echo "Building and running tests..."
	@for f in tests/*.c; do \
	    [ -f "$$f" ] || continue; \
	    name=$$(basename $$f .c); \
	    $(CC) $(CFLAGS) $$f $(OBJS) -o $(BIN_DIR)/$$name $(LDFLAGS) && \
	    echo "PASS: $$name" || echo "FAIL: $$name"; \
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
