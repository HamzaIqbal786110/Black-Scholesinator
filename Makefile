CC=gcc
CFLAGS=-Wall -Wextra -std=c11 -lm

all: black_scholes_base monte_carlo_base

black_scholes_base: black_scholes_base.c parser.c
	$(CC) black_scholes_base.c parser.c -o black_scholes_base $(CFLAGS)

monte_carlo_base: monte_carlo_base.c parser.c
	$(CC) monte_carlo_base.c parser.c -o monte_carlo_base $(CFLAGS)

clean:
	rm -f black_scholes_base monte_carlo_base
