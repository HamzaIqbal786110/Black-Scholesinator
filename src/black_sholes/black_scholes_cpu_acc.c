#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parser.h"
#include <omp.h>
#include <time.h>

#define IDX(i, j, p_steps) ((j) * ((p_steps) + 1) + (i))


double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// Struct that is used to help with data localization in individual option grids
typedef struct 
{
    double *c_vals;
    double *p_vals;
    double *c_a;
    double *c_b;
    double *c_c;
    double *p_a;
    double *p_b;
    double *p_c;
} fdm_struct;


int main() 
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
    fdm_struct *grid = (fdm_struct*) malloc(sizeof(fdm_struct) * count);
    for (int i = 0; i < count; i++) 
    {
        grid[i].c_vals = calloc((p_steps + 1) * (t_steps + 1), sizeof(double));
        grid[i].p_vals = calloc((p_steps + 1) * (t_steps + 1), sizeof(double));
        grid[i].c_a = calloc(p_steps + 1, sizeof(double));
        grid[i].c_b = calloc(p_steps + 1, sizeof(double));
        grid[i].c_c = calloc(p_steps + 1, sizeof(double));
        grid[i].p_a = calloc(p_steps + 1, sizeof(double));
        grid[i].p_b = calloc(p_steps + 1, sizeof(double));
        grid[i].p_c = calloc(p_steps + 1, sizeof(double));
    }

    start = CLOCK();
    // Loop over all options and run finite difference method (explicit scheme)
    int threads = (count < 48) ? count : 48;
    #pragma omp parallel for num_threads(threads)
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
            grid[opt].c_vals[IDX(i, t_steps, p_steps)] = fmax(s - k, 0.0);
            grid[opt].p_vals[IDX(i, t_steps, p_steps)] = fmax(k - s, 0.0);
            // c_vals[i][t_steps] = fmax(s - k, 0.0);
            // p_vals[i][t_steps] = fmax(k - s, 0.0);
            
            if (i > 0 && i < p_steps) 
            {
                double c_gamma = c_sig * c_sig * s * s;
                double p_gamma = p_sig * p_sig * s * s;
                double drift   = r * s;

                grid[opt].c_a[i] = 0.5 * dt * (c_gamma - drift) / (ds * ds);
                grid[opt].c_b[i] = 1.0 - dt * (c_gamma / (ds * ds) + r);
                grid[opt].c_c[i] = 0.5 * dt * (c_gamma + drift) / (ds * ds);

                grid[opt].p_a[i] = 0.5 * dt * (p_gamma - drift) / (ds * ds);
                grid[opt].p_b[i] = 1.0 - dt * (p_gamma / (ds * ds) + r);
                grid[opt].p_c[i] = 0.5 * dt * (p_gamma + drift) / (ds * ds);

            }
        }

        // Boundary conditions
        for (int j = 0; j <= t_steps; j++) 
        {
            double T = j * dt;
            // Call option boundaries
            grid[opt].c_vals[IDX(0, j, p_steps)] = 0.0;
            grid[opt].c_vals[IDX(p_steps, j, p_steps)] = s_max - k * exp(-r * (t - T));
            // c_vals[0][j] = 0.0;
            // c_vals[p_steps][j] = s_max - k * exp(-r * (t - T));
            
            // Put option boundaries
            grid[opt].p_vals[IDX(0, j, p_steps)] = k * exp(-r * (t - T));
            grid[opt].p_vals[IDX(p_steps, j, p_steps)] = 0.0;
            // p_vals[0][j] = k * exp(-r * (t - T));
            // p_vals[p_steps][j] = 0.0;
        }

        // Backward time stepping
        for (int j = t_steps - 1; j >= 0; j--) 
        {
            #pragma omp simd
            for (int i = 1; i < p_steps; i++) 
            {
                grid[opt].c_vals[IDX(i, j, p_steps)] =
                    grid[opt].c_a[i] * grid[opt].c_vals[IDX(i - 1, j + 1, p_steps)] +
                    grid[opt].c_b[i] * grid[opt].c_vals[IDX(i,     j + 1, p_steps)] +
                    grid[opt].c_c[i] * grid[opt].c_vals[IDX(i + 1, j + 1, p_steps)];

                grid[opt].p_vals[IDX(i, j, p_steps)] =
                    grid[opt].p_a[i] * grid[opt].p_vals[IDX(i - 1, j + 1, p_steps)] +
                    grid[opt].p_b[i] * grid[opt].p_vals[IDX(i,     j + 1, p_steps)] +
                    grid[opt].p_c[i] * grid[opt].p_vals[IDX(i + 1, j + 1, p_steps)];
            }
        }

        // Interpolate price at current stock price
        int s0_ind = (int)((stock - s_min) / ds);
        call_prices[opt] = grid[opt].c_vals[IDX(s0_ind, 0, p_steps)];
        put_prices[opt]  = grid[opt].p_vals[IDX(s0_ind, 0, p_steps)];
    }
    end = CLOCK();
    total = end - start;

    // Display results
    printf("\n CPU ACCELERATED IMPLEMENTATION\n");
    printf("--------------------------------------------------------------\n");
    printf("\n%-6s | %-10s | %-10s | %-10s | %-10s\n", "Index", "Call Model", "Call Actual", "Put Model", "Put Actual");
    printf("--------------------------------------------------------------\n");
    for (int i = 0; i < count; i++) 
    {
        printf("%-6d | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n",
               i, call_prices[i], options[i].c_mid, put_prices[i], options[i].p_mid);
    }
    printf("\nTIME TAKEN = %.5f ms\n", total);

    for (int i = 0; i < count; i++) 
    {
        free(grid[i].c_vals);
        free(grid[i].p_vals);
        free(grid[i].c_a);
        free(grid[i].c_b);
        free(grid[i].c_c);
        free(grid[i].p_a);
        free(grid[i].p_b);
        free(grid[i].p_c);
    }
    free(grid);  // Free the array of pointers
    free(call_prices); free(put_prices);
    free(options);

    return 0;
}
