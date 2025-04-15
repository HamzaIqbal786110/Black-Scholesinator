#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>


#define MAX_LINE_LENGTH 1024
#define MAX_LINES 10

int N_ITER;
int N_RUNS;

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

option_spread *read_csv(const char *filename, int *count) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int idx = 0;

    option_spread *options = (option_spread*) malloc(MAX_LINES * sizeof(option_spread));
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
                case 10: opt.c_iv	= value; break;
                case 12: opt.c_mid	= value; break;
                case 18: opt.p_iv	= value; break;
                case 20: opt.p_mid	= value; break;
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
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

__global__ void init_random(curandState *states, unsigned long seed, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iters) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void monte_carlo_kernel(double *block_prices, curandState *states, double s0, double r, double sigma_c, double sigma_p, double T, double k, int iters){

    extern __shared__ double shared_prices[]; //size dynamic based on N
    double local_prices[2] = {0};

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x ;
    if (idx >= iters) return;

    curandState local_state = states[idx];

    double u1 = curand_uniform_double(&local_state);
    double u2 = curand_uniform_double(&local_state);
    double z_c = sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);

    u1 = curand_uniform_double(&local_state);
    u2 = curand_uniform_double(&local_state);
    double z_p = sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);

    // Simulate for z_c and -z_c (call side)
    double s_c_pos = s0 * exp(((r - 0.5 * sigma_c * sigma_c) * T) + sigma_c * sqrt(T) * z_c);
    double s_c_neg = s0 * exp(((r - 0.5 * sigma_c * sigma_c) * T) + sigma_c * sqrt(T) * -z_c);
    local_prices[0] = 0.5 * (fmax(s_c_pos - k, 0.0) + fmax(s_c_neg - k, 0.0));

    // Simulate for z_p and -z_p (put side)
    double s_p_pos = s0 * exp(((r - 0.5 * sigma_p * sigma_p) * T) + sigma_p * sqrt(T) * z_p);
    double s_p_neg = s0 * exp(((r - 0.5 * sigma_p * sigma_p) * T) + sigma_p * sqrt(T) * -z_p);
    local_prices[1] = 0.5 * (fmax(k - s_p_pos, 0.0) + fmax(k - s_p_neg, 0.0));

    shared_prices[2 * tid + 0] = local_prices[0];
    shared_prices[2 * tid + 1] = local_prices[1];

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>=1){
        if (tid < stride){
            shared_prices[2 * tid + 0] += shared_prices[2 * (tid+stride) + 0];
            shared_prices[2 * tid + 1] += shared_prices[2 * (tid+stride) + 1];
        }
        __syncthreads();
    }

    if (tid == 0){
        block_prices[2 * blockIdx.x + 0] = shared_prices[0];
        block_prices[2 * blockIdx.x + 1] = shared_prices[1];
    }
}

void monte_carlo_pricer(option_spread option, double prices[2]){

    double s0 = option.underlying;
    double r = option.rfr;
    const double sigma_c = option.c_iv;
    const double sigma_p = option.p_iv;
    double T = option.dte / 365.0;
    double k = option.strike;

    int block_size = 256;
    int blocks = (N_ITER + block_size -1) / block_size;
    int shared_mem_size = 2 * block_size * sizeof(double);

    double *d_block_prices;
    double *h_block_prices = (double*)malloc(2 * blocks * sizeof(double));
    cudaMalloc(&d_block_prices, 2 * blocks * sizeof(double));

    curandState *d_states;
    cudaMalloc(&d_states, N_ITER * sizeof(curandState));
    init_random<<<blocks, block_size>>>(d_states, time(NULL), N_ITER);

    monte_carlo_kernel<<<blocks, block_size, shared_mem_size>>>(d_block_prices, d_states, s0, r, sigma_c, sigma_p, T, k, N_ITER);

    cudaMemcpy(h_block_prices, d_block_prices, 2 * blocks * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; i++){
        prices[0] += h_block_prices[2 * i + 0];
        prices[1] += h_block_prices[2 * i + 1];
    }
    cudaFree(d_block_prices);
    cudaFree(d_states);

    return;
}

int main(int argc, char **argv){

    if (argc != 3) {
        printf("Usage: %s <num_iterations> <num_runs>\n", argv[0]);
        return 1;
    }
    
    N_ITER = atoi(argv[1]);
    N_RUNS = atoi(argv[2]);

    double total_time = 0;
    for (int runs = 0; runs <N_RUNS; runs++){
        int count = 0;
        option_spread *options = read_csv("nvda_data_filtered.csv", &count);

        double start_time = CLOCK();

        for (int i = 0; i < count; i++) {
            double prices[2];
            monte_carlo_pricer(options[i], prices);
        }

        free(options);

    double end_time = CLOCK();
    total_time += end_time - start_time;
    
    }

    printf("AVG Time: %f\n", total_time/N_RUNS);
    return 0;
}
