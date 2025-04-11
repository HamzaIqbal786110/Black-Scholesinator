# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11
LDFLAGS = -lm -lrt

# OpenMP flag for CPU accelerated code
OMPFLAGS = -fopenmp

# Directories
SRC_DIR = src
BIN_DIR = bin
INCLUDE_DIR = include

# Source files
BS_BASE_SRC = $(SRC_DIR)/black_sholes/black_scholes_base.c
MC_BASE_SRC = $(SRC_DIR)/monte_carlo/monte_carlo_base.c
BS_CPU_ACC_SRC = $(SRC_DIR)/black_sholes/black_scholes_cpu_acc.c
COMMON_SRC = $(SRC_DIR)/common/parser.c

# Output binaries
BS_BASE_BIN = $(BIN_DIR)/black_scholes_base
MC_BASE_BIN = $(BIN_DIR)/monte_carlo_base
BS_CPU_ACC_BIN = $(BIN_DIR)/black_scholes_cpu_acc

# Targets
all: $(BS_BASE_BIN) $(MC_BASE_BIN) $(BS_CPU_ACC_BIN)

$(BS_BASE_BIN): $(BS_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(BS_BASE_SRC) $(COMMON_SRC) -I$(INCLUDE_DIR) -o $@ $(LDFLAGS)

$(MC_BASE_BIN): $(MC_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(MC_BASE_SRC) $(COMMON_SRC) -I$(INCLUDE_DIR) -o $@ $(LDFLAGS)

$(BS_CPU_ACC_BIN): $(BS_CPU_ACC_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(BS_CPU_ACC_SRC) $(COMMON_SRC) -I$(INCLUDE_DIR) -o $@ $(LDFLAGS)

clean:
	rm -f $(BIN_DIR)/*
