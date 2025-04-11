#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "parser.h"

#define IDX(i, j, p_steps) ((j) * ((p_steps) + 1) + (i))

double CLOCK() 
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// CUDA kernel for backward time stepping
__global__ void fdm_kernel(
    double *c_vals, double *p_vals,
    const double *c_a, const double *c_b, const double *c_c,
    const double *p_a, const double *p_b, const double *p_c,
    int p_steps, int t_steps, int count)
{
    int opt_id = blockIdx.x;
    int i = threadIdx.x + 1;

    if (opt_id >= count || i >= p_steps) return;

    size_t grid_size = (p_steps + 1) * (t_steps + 1);
    size_t coef_offset = opt_id * (p_steps + 1);
    size_t offset = opt_id * grid_size;

    for (int j = t_steps - 1; j >= 0; j--) 
    {
        size_t idx   = offset + IDX(i, j, p_steps);
        size_t idx_l = offset + IDX(i - 1, j + 1, p_steps);
        size_t idx_c = offset + IDX(i,     j + 1, p_steps);
        size_t idx_r = offset + IDX(i + 1, j + 1, p_steps);

        c_vals[idx] =
            c_a[coef_offset + i] * c_vals[idx_l] +
            c_b[coef_offset + i] * c_vals[idx_c] +
            c_c[coef_offset + i] * c_vals[idx_r];

        p_vals[idx] =
            p_a[coef_offset + i] * p_vals[idx_l] +
            p_b[coef_offset + i] * p_vals[idx_c] +
            p_c[coef_offset + i] * p_vals[idx_r];
    }
}

int main() 
{
    int count = 0;
    option_spread *options_host = read_csv("Data/nvda_data_filtered.csv", &count);

    const int p_steps = 200;
    const int t_steps = 100000;
    size_t grid_size = (p_steps + 1) * (t_steps + 1);
    size_t grid_bytes = sizeof(double) * grid_size;

    double *call_vals_host = (double *)calloc(grid_size * count, sizeof(double));
    double *put_vals_host  = (double *)calloc(grid_size * count, sizeof(double));
    double *call_prices_host = (double *)malloc(sizeof(double) * count);
    double *put_prices_host  = (double *)malloc(sizeof(double) * count);

    double *c_a = (double *)malloc(sizeof(double) * (p_steps + 1) * count);
    double *c_b = (double *)malloc(sizeof(double) * (p_steps + 1) * count);
    double *c_c = (double *)malloc(sizeof(double) * (p_steps + 1) * count);
    double *p_a = (double *)malloc(sizeof(double) * (p_steps + 1) * count);
    double *p_b = (double *)malloc(sizeof(double) * (p_steps + 1) * count);
    double *p_c = (double *)malloc(sizeof(double) * (p_steps + 1) * count);

    double start = CLOCK();
    for (int opt = 0; opt < count; opt++) 
    {
        double s0 = options_host[opt].underlying;
        double k = options_host[opt].strike;
        double r = options_host[opt].rfr;
        double sigma_c = options_host[opt].c_iv;
        double sigma_p = options_host[opt].p_iv;
        double T = options_host[opt].dte / 365.0;
        double ds = (3.0 * s0 - 0.5 * s0) / p_steps;
        double dt = T / t_steps;
        double s_min = 0.5 * s0;
        size_t offset = opt * grid_size;
        size_t coef_offset = opt * (p_steps + 1);

        for (int i = 0; i <= p_steps; i++) 
        {
            double s = s_min + i * ds;
            call_vals_host[offset + IDX(i, t_steps, p_steps)] = fmax(s - k, 0.0);
            put_vals_host[offset + IDX(i, t_steps, p_steps)]  = fmax(k - s, 0.0);

            if (i > 0 && i < p_steps) 
            {
                double gamma_c = sigma_c * sigma_c * s * s;
                double gamma_p = sigma_p * sigma_p * s * s;
                double drift = r * s;

                c_a[coef_offset + i] = 0.5 * dt * (gamma_c - drift) / (ds * ds);
                c_b[coef_offset + i] = 1.0 - dt * (gamma_c / (ds * ds) + r);
                c_c[coef_offset + i] = 0.5 * dt * (gamma_c + drift) / (ds * ds);

                p_a[coef_offset + i] = 0.5 * dt * (gamma_p - drift) / (ds * ds);
                p_b[coef_offset + i] = 1.0 - dt * (gamma_p / (ds * ds) + r);
                p_c[coef_offset + i] = 0.5 * dt * (gamma_p + drift) / (ds * ds);
            }
        }

        for (int j = 0; j <= t_steps; j++) 
        {
            double T_j = j * dt;
            call_vals_host[offset + IDX(0, j, p_steps)] = 0.0;
            call_vals_host[offset + IDX(p_steps, j, p_steps)] = (3.0 * s0) - k * exp(-r * (T - T_j));
            put_vals_host[offset + IDX(0, j, p_steps)] = k * exp(-r * (T - T_j));
            put_vals_host[offset + IDX(p_steps, j, p_steps)] = 0.0;
        }
    }

    // Device memory
    double *call_vals_dev, *put_vals_dev;
    double *c_a_dev, *c_b_dev, *c_c_dev;
    double *p_a_dev, *p_b_dev, *p_c_dev;

    cudaMalloc(&call_vals_dev, grid_bytes * count);
    cudaMalloc(&put_vals_dev,  grid_bytes * count);
    cudaMalloc(&c_a_dev, sizeof(double) * (p_steps + 1) * count);
    cudaMalloc(&c_b_dev, sizeof(double) * (p_steps + 1) * count);
    cudaMalloc(&c_c_dev, sizeof(double) * (p_steps + 1) * count);
    cudaMalloc(&p_a_dev, sizeof(double) * (p_steps + 1) * count);
    cudaMalloc(&p_b_dev, sizeof(double) * (p_steps + 1) * count);
    cudaMalloc(&p_c_dev, sizeof(double) * (p_steps + 1) * count);

    cudaMemcpy(call_vals_dev, call_vals_host, grid_bytes * count, cudaMemcpyHostToDevice);
    cudaMemcpy(put_vals_dev,  put_vals_host,  grid_bytes * count, cudaMemcpyHostToDevice);
    cudaMemcpy(c_a_dev, c_a, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(c_b_dev, c_b, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(c_c_dev, c_c, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(p_a_dev, p_a, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(p_b_dev, p_b, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(p_c_dev, p_c, sizeof(double) * (p_steps + 1) * count, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 gridDim(count);
    dim3 blockDim(p_steps);
    fdm_kernel<<<gridDim, blockDim>>>(call_vals_dev, put_vals_dev, c_a_dev, c_b_dev, c_c_dev,
                                      p_a_dev, p_b_dev, p_c_dev, p_steps, t_steps, count);
    cudaDeviceSynchronize();

    // Copy final prices back
    for (int opt = 0; opt < count; opt++) 
    {
        int s0_index = (int)((options_host[opt].underlying - 0.5 * options_host[opt].underlying) / ((3.0 - 0.5) * options_host[opt].underlying / p_steps));
        cudaMemcpy(&call_prices_host[opt], call_vals_dev + opt * grid_size + IDX(s0_index, 0, p_steps), sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&put_prices_host[opt],  put_vals_dev  + opt * grid_size + IDX(s0_index, 0, p_steps), sizeof(double), cudaMemcpyDeviceToHost);
    }

    double time_taken = CLOCK() - start;

    // Output
    printf("\n GPU ACCELERATED IMPLEMENTATION\n");
    printf("--------------------------------------------------------------\n");
    printf("\n%-6s | %-10s | %-10s | %-10s | %-10s\n", "Index", "Call Model", "Call Actual", "Put Model", "Put Actual");
    printf("--------------------------------------------------------------\n");
    for (int i = 0; i < count; i++) {
        printf("%-6d | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n",
               i, call_prices_host[i], options_host[i].c_mid, put_prices_host[i], options_host[i].p_mid);
    }
    printf("\nTIME TAKEN = %.5f ms\n", time_taken);
    // Free everything
    free(options_host);
    free(call_prices_host);
    free(put_prices_host);
    free(call_vals_host);
    free(put_vals_host);
    free(c_a); free(c_b); free(c_c);
    free(p_a); free(p_b); free(p_c);
    cudaFree(call_vals_dev); cudaFree(put_vals_dev);
    cudaFree(c_a_dev); cudaFree(c_b_dev); cudaFree(c_c_dev);
    cudaFree(p_a_dev); cudaFree(p_b_dev); cudaFree(p_c_dev);

    return 0;
}
