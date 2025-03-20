#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define INITIAL_CAPACITY 10000

typedef struct {
    double time;
    double underlying;
    double expire_time;
    double dte;
    double strike;
    double c_delta;
    double c_gamma;
    double c_vega;
    double c_theta;
    double c_rho;
    double c_iv;
    double c_volume;
    double c_mid;
    double p_delta;
    double p_gamma;
    double p_vega;
    double p_theta;
    double p_rho;
    double p_iv;
    double p_volume;
    double p_mid;
    double rfr;
} option_spread;

option_spread *read_csv(const char *filename, int *count) 
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LENGTH];
    int capacity = INITIAL_CAPACITY;
    int idx = 0;

    option_spread *options = malloc(capacity * sizeof(option_spread));
    if (!options) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // Skip header
    fgets(line, sizeof(line), file);

    // Read each line
    while (fgets(line, sizeof(line), file)) {
        if (idx >= capacity) {
            capacity *= 2;
            options = realloc(options, capacity * sizeof(option_spread));
            if (!options) {
                perror("Memory reallocation failed");
                exit(EXIT_FAILURE);
            }
        }

        option_spread opt = {0}; // Initialize all fields to zero
        char *token;
        int field = 0;

        token = strtok(line, ",");
        while (token && field < 22) {
            if (*token == '\0') {
                // Handle missing values
                token = strtok(NULL, ",");
                field++;
                continue;
            }
            
            switch (field) {
                case 0: opt.time = atof(token); break;
                case 1: opt.underlying = atof(token); break;
                case 2: opt.expire_time = atof(token); break;
                case 3: opt.dte = atof(token); break;
                case 4: opt.strike = atof(token); break;
                case 5: opt.c_delta = atof(token); break;
                case 6: opt.c_gamma = atof(token); break;
                case 7: opt.c_vega = atof(token); break;
                case 8: opt.c_theta = atof(token); break;
                case 9: opt.c_rho = atof(token); break;
                case 10: opt.c_iv = atof(token); break;
                case 11: opt.c_volume = atof(token); break;
                case 12: opt.c_mid = atof(token); break;
                case 13: opt.p_delta = atof(token); break;
                case 14: opt.p_gamma = atof(token); break;
                case 15: opt.p_vega = atof(token); break;
                case 16: opt.p_theta = atof(token); break;
                case 17: opt.p_rho = atof(token); break;
                case 18: opt.p_iv = atof(token); break;
                case 19: opt.p_volume = atof(token); break;
                case 20: opt.p_mid = atof(token); break;
                case 21: opt.rfr = atof(token); break;
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

int main() 
{
    int count = 0;
    option_spread *options = read_csv("Data/nvda_data.csv", &count);

    for (int i = 0; i < count && i < 5; i++) {
        printf("Entry %d:\n"
               "  Time: %.0f, Underlying: %.2f, Expire Time: %.0f, DTE: %.2f, Strike: %.2f\n"
               "  Call - Delta: %.6f, Gamma: %.6f, Vega: %.6f, Theta: %.6f, Rho: %.6f, IV: %.6f, Volume: %.2f, Mid: %.6f\n"
               "  Put  - Delta: %.6f, Gamma: %.6f, Vega: %.6f, Theta: %.6f, Rho: %.6f, IV: %.6f, Volume: %.2f, Mid: %.6f\n"
               "  Risk-Free Rate: %.6f\n",
               i, options[i].time, options[i].underlying, options[i].expire_time, options[i].dte, options[i].strike,
               options[i].c_delta, options[i].c_gamma, options[i].c_vega, options[i].c_theta, options[i].c_rho, options[i].c_iv, options[i].c_volume, options[i].c_mid,
               options[i].p_delta, options[i].p_gamma, options[i].p_vega, options[i].p_theta, options[i].p_rho, options[i].p_iv, options[i].p_volume, options[i].p_mid,
               options[i].rfr);
    }

    // Initial Pricing Strategy Implementation
    float s_min = 0;
    float s_max, k, sig, r, t, ds, dt, s;
    int p_steps = 1000;
    int t_steps = 10000;
    float **c_vals = (float*)calloc((p_steps + 1), sizeof(float));
    float **p_vals = (float*)calloc((p_steps + 1), sizeof(float));
    float *a = (float*)calloc((p_stps + 1), sizeof(float));
    float *b = (float*)calloc((p_stps + 1), sizeof(float));
    float *c = (float*)calloc((p_stps + 1), sizeof(float));
    for(int j = 0; j <= p_steps; j++)
    {
        c_vals[j] = (float*)calloc((t_steps + 1), sizeof(float));
        p_vals[j] = (float*)calloc((t_steps + 1), sizeof(float));
    }

    for(int opt = 0; opt < count; opt++)
    {
        s_max = options[opt].underlying;
        k = options[opt].strike;
        sig = options[opt].iv;
        r = options[opt].rfr;
        t = options[opt].dte / 365;
        ds = (s_max - s_min) / p_steps;
        dt = t / t_steps;

        for(int i = 0; i <= p_steps; i++)
        {
            s = i * ds;
            c_vals[i][t_steps] = (s - k >= 0) ? (s - k) : 0;
            p_vals[i][t_steps] = (k - s <= 0) ? (k - s) : 0;
            if(i > 0 && i < p_steps)
            {
                a[i] = 0.5 * dt *(((sig*sig) * (i*i)) - (r * i));
                b[i] = 1 - (dt * ((sig*sig * i*i) + r));
                c[i] = (0.5 * dt) * ((sig*sig * i*i) + (r * i));
            }
        }


    }

    free(options);
    return 0;
}
