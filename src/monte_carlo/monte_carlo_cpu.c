#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

double gaussian_random(unsigned int* seed) 
{
    double u1 = ((double) rand_r(seed) + 1.0) / ((double) RAND_MAX + 2.0); // avoid log(0)
    double u2 = ((double) rand_r(seed) + 1.0) / ((double) RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);
}

void monte_carlo_pricer(option_spread option, double prices[2])
{
    unsigned int seed;   
    #pragma omp parallel
{
    seed = omp_get_thread_num() + time(NULL);
}

    double s0 = option.underlying;
    double r = option.rfr;
    const double sigma_c = option.c_iv;
    const double sigma_p = option.p_iv;
    double T = option.dte / 365.0;
    double k = option.strike;
    double z_c, z_p, s_c_pos, s_c_neg, s_p_pos, s_p_neg;
    double v_c = 0, v_p = 0;

    #pragma omp parallel for reduction(+:v_c,v_p) private(z_c, z_p, s_c_pos, s_c_neg, s_p_pos, s_p_neg)
    for(int i = 0; i < N_ITER; i++)
    {
        z_c = gaussian_random(&seed);
        z_p = gaussian_random(&seed);

        // Simulate for z_c and -z_c (call side)
        s_c_pos = s0 * exp(((r - 0.5 * sigma_c * sigma_c) * T) + sigma_c * sqrt(T) * z_c);
        s_c_neg = s0 * exp(((r - 0.5 * sigma_c * sigma_c) * T) + sigma_c * sqrt(T) * -z_c);
        v_c += 0.5 * (fmax(s_c_pos - k, 0) + fmax(s_c_neg - k, 0));

        // Simulate for z_p and -z_p (put side)
        s_p_pos = s0 * exp(((r - 0.5 * sigma_p * sigma_p) * T) + sigma_p * sqrt(T) * z_p);
        s_p_neg = s0 * exp(((r - 0.5 * sigma_p * sigma_p) * T) + sigma_p * sqrt(T) * -z_p);
        v_p += 0.5 * (fmax(k - s_p_pos, 0) + fmax(k - s_p_neg, 0));

    }
    v_c = exp(-r * T) * v_c / N_ITER;
    v_p = exp(-r * T) * v_p / N_ITER;
    prices[0] = v_c;
    prices[1] = v_p;

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
   // printf("Index  | Call Model | Call Actual | Put Model  | Put Actual\n");
    //printf("--------------------------------------------------------------\n");

    double start_time = CLOCK();

  
    for (int i = 0; i < count; i++) {
        double prices[2];
        monte_carlo_pricer(options[i], prices);
        //printf("%-7d| %-11.4f| %-12.4f| %-11.4f| %-10.4f\n",
          //     i,
            //   prices[0], options[i].c_mid,
              // prices[1], options[i].p_mid);
    }

    free(options);

    double end_time = CLOCK();
    total_time += end_time - start_time;

    }
    
    printf("AVG Time: %f\n", total_time/N_RUNS);
    return 0;
}
