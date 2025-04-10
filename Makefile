CC=gcc
CFLAGS=-Wall -Wextra -std=c11 -lm

all: black_scholes monte_carlo

black_scholes: black_scholes.c parser.c
	$(CC) black_scholes.c parser.c -o black_scholes $(CFLAGS)

monte_carlo: monte_carlo.c parser.c
	$(CC) monte_carlo.c parser.c -o monte_carlo $(CFLAGS)

clean:
	rm -f black_scholes monte_carlo
