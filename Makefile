# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -fopenmp
NVCC = nvcc
NVCCFLAGS = -Xcompiler "-fopenmp -O3"
INCLUDES = -Iinclude

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Source files
COMMON_SRC = $(SRC_DIR)/common/parser.c
BLACK_SCHOLES_BASE_SRC = $(SRC_DIR)/black_sholes/black_scholes_base.c
BLACK_SCHOLES_CPU_ACC_SRC = $(SRC_DIR)/black_sholes/black_scholes_cpu_acc.c
BLACK_SCHOLES_GPU_SRC = $(SRC_DIR)/black_sholes/black_sholes_gpu.cu
MONTE_CARLO_BASE_SRC = $(SRC_DIR)/monte_carlo/monte_carlo_base.c

# Object files
COMMON_OBJ = $(OBJ_DIR)/parser.o

# Binaries
BLACK_SCHOLES_BASE_BIN = $(BIN_DIR)/black_scholes_base
BLACK_SCHOLES_CPU_ACC_BIN = $(BIN_DIR)/black_scholes_cpu_acc
BLACK_SCHOLES_GPU_BIN = $(BIN_DIR)/black_scholes_gpu
MONTE_CARLO_BASE_BIN = $(BIN_DIR)/monte_carlo_base

# Default target
all: $(BLACK_SCHOLES_BASE_BIN) $(BLACK_SCHOLES_CPU_ACC_BIN) $(BLACK_SCHOLES_GPU_BIN) $(MONTE_CARLO_BASE_BIN)

# Compile C source to object
$(COMMON_OBJ): $(COMMON_SRC) include/parser.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Base CPU binary
$(BLACK_SCHOLES_BASE_BIN): $(BLACK_SCHOLES_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lm -lrt

# Accelerated CPU binary
$(BLACK_SCHOLES_CPU_ACC_BIN): $(BLACK_SCHOLES_CPU_ACC_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lm -lrt

# GPU binary (linking precompiled parser.o)
$(BLACK_SCHOLES_GPU_BIN): $(BLACK_SCHOLES_GPU_SRC) $(COMMON_OBJ)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< obj/parser.o

# Monte Carlo base binary
$(MONTE_CARLO_BASE_BIN): $(MONTE_CARLO_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lm

# Create directories if they don't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean up build artifacts
clean:
	rm -f $(BIN_DIR)/*
	rm -f $(OBJ_DIR)/*.o
