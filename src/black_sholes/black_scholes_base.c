#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "parser.h"

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int main(int argc, char *argv[]) 
{
    double start, end, total;
    int count = 0;
    option_spread *options = read_csv("Data/nvda_data_filtered.csv", &count);

    // Allocate memory for model price outputs
    double *call_prices = malloc(sizeof(double) * count);
    double *put_prices  = malloc(sizeof(double) * count);

    // FDM grid configuration
    int p_steps = 200;
    int t_steps = 100000;
    if (argc >= 2) 
    {
        p_steps = atoi(argv[1]);
    }

    if (argc >= 3) 
    {
        t_steps = atoi(argv[2]);
    }

    double **c_vals = calloc(p_steps + 1, sizeof(double*));
    double **p_vals = calloc(p_steps + 1, sizeof(double*));
    double *c_a = calloc(p_steps + 1, sizeof(double));
    double *c_b = calloc(p_steps + 1, sizeof(double));
    double *c_c = calloc(p_steps + 1, sizeof(double));
    double *p_a = calloc(p_steps + 1, sizeof(double));
    double *p_b = calloc(p_steps + 1, sizeof(double));
    double *p_c = calloc(p_steps + 1, sizeof(double));

    for (int j = 0; j <= p_steps; j++) {
        c_vals[j] = calloc(t_steps + 1, sizeof(double));
        p_vals[j] = calloc(t_steps + 1, sizeof(double));
    }

    start = CLOCK();
    // Loop over all options and run finite difference method (explicit scheme)
    for (int opt = 0; opt < count; opt++) 
    {
        double stock = options[opt].underlying;
        double k     = options[opt].strike;
        double c_sig = options[opt].c_iv;
        double p_sig = options[opt].p_iv;
        double r     = options[opt].rfr;
        double t     = options[opt].dte / 365.0;

        double s_min = 0.5 * stock;
        double s_max = 3.0 * stock;
        double ds    = (s_max - s_min) / p_steps;
        double dt    = t / t_steps;

        // Terminal condition and coefficient setup
        for (int i = 0; i <= p_steps; i++) 
        {
            double s = s_min + i * ds;
            c_vals[i][t_steps] = fmax(s - k, 0.0);
            p_vals[i][t_steps] = fmax(k - s, 0.0);
            
            if (i > 0 && i < p_steps) 
            {
                double c_gamma = c_sig * c_sig * s * s;
                double p_gamma = p_sig * p_sig * s * s;
                double drift   = r * s;

                c_a[i] = 0.5 * dt * (c_gamma - drift) / (ds * ds);
                c_b[i] = 1.0 - dt * (c_gamma / (ds * ds) + r);
                c_c[i] = 0.5 * dt * (c_gamma + drift) / (ds * ds);

                p_a[i] = 0.5 * dt * (p_gamma - drift) / (ds * ds);
                p_b[i] = 1.0 - dt * (p_gamma / (ds * ds) + r);
                p_c[i] = 0.5 * dt * (p_gamma + drift) / (ds * ds);

            }
        }

        // Boundary conditions
        for (int j = 0; j <= t_steps; j++) 
        {
            double T = j * dt;
            // Call option boundaries
            c_vals[0][j] = 0.0;
            c_vals[p_steps][j] = s_max - k * exp(-r * (t - T));
            // Put option boundaries
            p_vals[0][j] = k * exp(-r * (t - T));
            p_vals[p_steps][j] = 0.0;
        }

        // Backward time stepping
        for (int j = t_steps - 1; j >= 0; j--) {
            for (int i = 1; i < p_steps; i++) {
                c_vals[i][j] = c_a[i] * c_vals[i-1][j+1] + c_b[i] * c_vals[i][j+1] + c_c[i] * c_vals[i+1][j+1];
                p_vals[i][j] = p_a[i] * p_vals[i-1][j+1] + p_b[i] * p_vals[i][j+1] + p_c[i] * p_vals[i+1][j+1];
            }
        }

        // Interpolate price at current stock price
        int s0_ind = (int)((stock - s_min) / ds);
        call_prices[opt] = c_vals[s0_ind][0];
        put_prices[opt]  = p_vals[s0_ind][0];
    }
    end = CLOCK();

    // Display results
    printf("\nBASIC IMPLEMENTATION\n");
    printf("--------------------------------------------------------------\n");
    printf("\n%-6s | %-10s | %-10s | %-10s | %-10s\n", "Index", "Call Model", "Call Actual", "Put Model", "Put Actual");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < count; i++) {
        printf("%-6d | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n",
               i, call_prices[i], options[i].c_mid, put_prices[i], options[i].p_mid);
    }

    total = end - start;
    printf("\nPSTEPS = %i, TSTEPS = %i\n", p_steps, t_steps);
    printf("\nTIME TAKEN = %.5f ms\n", total);
    // Free allocated memory
    for (int j = 0; j <= p_steps; j++) {
        free(c_vals[j]);
        free(p_vals[j]);
    }

    free(c_vals); free(p_vals);
    free(c_a); free(c_b); free(c_c);
    free(p_a); free(p_b); free(p_c);
    free(call_prices); free(put_prices);
    free(options);

    return 0;
}
