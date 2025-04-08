CC=gcc
CFLAGS=-Wall -Wextra -std=c11 -lm

all: black_scholes

black_scholes: black_scholes.c parser.c
	$(CC) black_scholes.c parser.c -o black_scholes $(CFLAGS)
