// csv_reader.h
#ifndef PARSER_H
#define PARSER_H

#define MAX_LINE_LENGTH 1024
#define MAX_LINES 10

typedef struct {
    double underlying;
    double strike;
    double dte;
    double c_iv;
    double c_mid;
    double p_iv;
    double p_mid;
    double rfr;
} option_spread;

option_spread *read_csv(const char *filename, int *count);

#endif