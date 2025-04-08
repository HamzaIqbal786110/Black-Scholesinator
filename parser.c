#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parser.h"

option_spread *read_csv(const char *filename, int *count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int idx = 0;

    option_spread *options = malloc(MAX_LINES * sizeof(option_spread));
    if (!options) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Skip header line
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && idx < MAX_LINES) {
        option_spread opt = {0};
        char *token = strtok(line, ",");
        int field = 0;

        while (token && field <= 21) {
            double value = (*token == '\0' || strcmp(token, "\n") == 0) ? NAN : atof(token);

            switch (field) {
                case 1:  opt.underlying = value; break;
                case 3:  opt.dte        = value; break;
                case 4:  opt.strike     = value; break;
                case 10: opt.c_iv       = value; break;
                case 12: opt.c_mid      = value; break;
                case 18: opt.p_iv       = value; break;
                case 20: opt.p_mid      = value; break;
                case 21: opt.rfr        = value; break;
            }

            token = strtok(NULL, ",");
            field++;
        }

        options[idx++] = opt;
    }

    fclose(file);
    *count = idx;
    return options;
}