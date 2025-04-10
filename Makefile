# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude
LDFLAGS = -lm

# Binary output directory
BIN_DIR = bin

# Source files
COMMON_SRC = src/common/parser.c
BS_BASE_SRC = src/black_sholes/black_scholes_base.c
MC_BASE_SRC = src/monte_carlo/monte_carlo_base.c

# Binaries
BS_BASE_BIN = $(BIN_DIR)/black_scholes_base
MC_BASE_BIN = $(BIN_DIR)/monte_carlo_base

# Default target
all: $(BS_BASE_BIN) $(MC_BASE_BIN)

# Black-Scholes base build
$(BS_BASE_BIN): $(BS_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Monte Carlo base build
$(MC_BASE_BIN): $(MC_BASE_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Clean build outputs
clean:
	rm -f $(BIN_DIR)/*
