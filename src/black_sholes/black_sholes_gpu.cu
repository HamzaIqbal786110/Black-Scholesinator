#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "parser.h"

#define IDX(i, j, price_steps) ((j) * ((price_steps) + 1) + (i))

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
    int price_steps, int time_steps, int count)
{
    int opt_id = blockIdx.x;
    int i = threadIdx.x + 1;

    if (opt_id >= count || i >= price_steps) return;

    size_t grid_size = (price_steps + 1) * (time_steps + 1);
    size_t coef_offset = opt_id * (price_steps + 1);
    size_t offset = opt_id * grid_size;

    for (int j = time_steps - 1; j >= 0; j--) 
    {
        size_t idx   = offset + IDX(i, j, price_steps);
        size_t idx_l = offset + IDX(i - 1, j + 1, price_steps);
        size_t idx_c = offset + IDX(i,     j + 1, price_steps);
        size_t idx_r = offset + IDX(i + 1, j + 1, price_steps);

        c_vals[idx] =
            c_a[coef_offset + i] * c_vals[idx_l] +
            c_b[coef_offset + i] * c_vals[idx_c] +
            c_c[coef_offset + i] * c_vals[idx_r];

        p_vals[idx] =
            p_a[coef_offset + i] * p_vals[idx_l] +
            p_b[coef_offset + i] * p_vals[idx_c] +
            p_c[coef_offset + i] * p_vals[idx_r];
        // if (opt_id == 2 && i == 40)
        // {
        //     printf("[FDM-BEGIN] opt=%d i=%d j=%d idx=%lu idx_l=%lu idx_c=%lu idx_r=%lu\n",
        //         opt_id, i, j, idx, idx_l, idx_c, idx_r);
        
        //     printf("[FDM-VALS] j=%d c_l=%.5f c_c=%.5f c_r=%.5f\n",
        //         j, c_vals[idx_l], c_vals[idx_c], c_vals[idx_r]);
        
        //     printf("[FDM-COEF] a=%.6f b=%.6f c=%.6f\n",
        //         c_a[coef_offset + i], c_b[coef_offset + i], c_c[coef_offset + i]);
        
        //     printf("[FDM-OUT] j=%d call=%.6f put=%.6f\n",
        //         j, c_vals[idx], p_vals[idx]);
        // }
    }
}

__global__ void setup_kernel(
    double *call_vals, double *put_vals,
    double *c_a, double *c_b, double *c_c,
    double *p_a, double *p_b, double *p_c,
    const option_spread *opts,
    int price_steps, int time_steps)
{
    int opt = blockIdx.x;
    int i = threadIdx.x;

    if (i > price_steps) return;

    option_spread o = opts[opt];

    double s0 = o.underlying;
    double k = o.strike;
    double r = o.rfr;
    double sigma_c = o.c_iv;
    double sigma_p = o.p_iv;
    double T = o.dte / 365.0;

    double ds = (3.0 * s0 - 0.5 * s0) / price_steps;
    double dt = T / time_steps;
    double s_min = 0.5 * s0;

    double s = s_min + i * ds;

    size_t grid_size = (price_steps + 1) * (time_steps + 1);
    size_t offset = opt * grid_size;
    size_t coef_offset = opt * (price_steps + 1);

    // terminal payoff at t = T
    call_vals[offset + IDX(i, time_steps, price_steps)] = fmax(s - k, 0.0);
    put_vals[offset + IDX(i, time_steps, price_steps)]  = fmax(k - s, 0.0);

    // coefficients (skip i=0 and i=price_steps)
    if (i > 0 && i < price_steps)
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
    if (opt == 2 && (i == 40 || i == 100))
    {
        printf("[SETUP] opt=%d i=%d s=%.4f ds=%.4f s_min=%.4f k=%.4f\n", opt, i, s, ds, s_min, k);
    }
}

__global__ void boundaries_kernel(
    double *call_vals, double *put_vals,
    const option_spread *opts,
    int price_steps, int time_steps)
{
    int opt = blockIdx.x;                              // option index
    int j = threadIdx.x + blockIdx.y * blockDim.x;     // time step index

    if (j > time_steps) return;

    option_spread o = opts[opt];
    double s0 = o.underlying;
    double k = o.strike;
    double r = o.rfr;
    double T = o.dte / 365.0;
    double dt = T / time_steps;
    double T_j = j * dt;

    size_t grid_size = (price_steps + 1) * (time_steps + 1);
    size_t offset = opt * grid_size;

    call_vals[offset + IDX(0, j, price_steps)] = 0.0;
    put_vals[offset + IDX(0, j, price_steps)] = k * exp(-r * (T - T_j));

    call_vals[offset + IDX(price_steps, j, price_steps)] = (3.0 * s0) - k * exp(-r * (T - T_j));
    put_vals[offset + IDX(price_steps, j, price_steps)] = 0.0;

    // if (opt == 2 && (j == 0 || j == time_steps))
    // {
    //     printf("[BOUNDARIES] opt=%d j=%d call_L=%.4f put_L=%.4f call_R=%.4f put_R=%.4f\n",
    //         opt, j,
    //         call_vals[offset + IDX(0, j, price_steps)],
    //         put_vals[offset + IDX(0, j, price_steps)],
    //         call_vals[offset + IDX(price_steps, j, price_steps)],
    //         put_vals[offset + IDX(price_steps, j, price_steps)]);
    // }
}

int main() 
{
    int count = 0;
    option_spread *options_host = read_csv("Data/nvda_data_filtered.csv", &count);

    const int price_steps = 200;
    const int time_steps = 100000;
    double *call_prices_host = (double *)malloc(sizeof(double) * count);
    double *put_prices_host  = (double *)malloc(sizeof(double) * count);

    double start = CLOCK();

    // === Device memory ===
    option_spread *options_gpu;
    double *call_vals_gpu, *put_vals_gpu;
    double *c_a_gpu, *c_b_gpu, *c_c_gpu;
    double *p_a_gpu, *p_b_gpu, *p_c_gpu;

    // Allocate options and upload
    cudaMalloc(&options_gpu, sizeof(option_spread) * count);
    cudaMemcpy(options_gpu, options_host, sizeof(option_spread) * count, cudaMemcpyHostToDevice);

    size_t grid_size = (price_steps + 1) * (time_steps + 1);
    size_t coeff_len = price_steps + 1;

    cudaMalloc(&call_vals_gpu, sizeof(double) * grid_size * count);
    cudaMalloc(&put_vals_gpu,  sizeof(double) * grid_size * count);
    cudaMemset(call_vals_gpu, 0, sizeof(double) * grid_size * count);
    cudaMemset(put_vals_gpu,  0, sizeof(double) * grid_size * count);


    cudaMalloc(&c_a_gpu, sizeof(double) * coeff_len * count);
    cudaMalloc(&c_b_gpu, sizeof(double) * coeff_len * count);
    cudaMalloc(&c_c_gpu, sizeof(double) * coeff_len * count);
    cudaMalloc(&p_a_gpu, sizeof(double) * coeff_len * count);
    cudaMalloc(&p_b_gpu, sizeof(double) * coeff_len * count);
    cudaMalloc(&p_c_gpu, sizeof(double) * coeff_len * count);

    // Launch setup kernel
    dim3 gridDim(count);
    dim3 blockDim(price_steps + 1);
    setup_kernel<<<gridDim, blockDim>>>(call_vals_gpu, put_vals_gpu, c_a_gpu, c_b_gpu, c_c_gpu, p_a_gpu, p_b_gpu, p_c_gpu, options_gpu, price_steps, time_steps);
    cudaDeviceSynchronize();

    // Launch boundaries kernel
    int block_size = 256;
    int num_blocks_y = (time_steps + block_size) / block_size;

    gridDim = dim3(count, num_blocks_y);  // x: options, y: time slices
    blockDim = dim3(block_size);          // threads per block

    boundaries_kernel<<<gridDim, blockDim>>>(call_vals_gpu, put_vals_gpu, options_gpu, price_steps, time_steps);
    cudaDeviceSynchronize();

    // Launch FDM solver kernel
    gridDim = dim3(count);
    blockDim = dim3(price_steps);
    fdm_kernel<<<gridDim, blockDim>>>(call_vals_gpu, put_vals_gpu, c_a_gpu, c_b_gpu, c_c_gpu, p_a_gpu, p_b_gpu, p_c_gpu, price_steps, time_steps, count);
    cudaDeviceSynchronize();

    // Copy final prices back
    for (int opt = 0; opt < count; opt++) 
    {
        double s0 = options_host[opt].underlying;
        double s_min = 0.5 * s0;
        double ds = (3.0 * s0 - s_min) / price_steps;
        int s0_index = (int)((s0 - s_min) / ds);        
        cudaMemcpy(&call_prices_host[opt], call_vals_gpu + opt * grid_size + IDX(s0_index, 0, price_steps), sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&put_prices_host[opt],  put_vals_gpu  + opt * grid_size + IDX(s0_index, 0, price_steps), sizeof(double), cudaMemcpyDeviceToHost);
        if (opt == 2)
        {
            printf("[MAIN-COPYBACK] opt=%d s0_index=%d call_val=%.6f put_val=%.6f\n",
                opt, s0_index, call_prices_host[opt], put_prices_host[opt]);
        }
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
    cudaFree(call_vals_gpu); cudaFree(put_vals_gpu);
    cudaFree(c_a_gpu); cudaFree(c_b_gpu); cudaFree(c_c_gpu);
    cudaFree(p_a_gpu); cudaFree(p_b_gpu); cudaFree(p_c_gpu);

    return 0;
}