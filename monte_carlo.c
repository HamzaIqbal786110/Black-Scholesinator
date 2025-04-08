#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "parser.h"

#define N_ITER 100000

double gaussian_random() 
{
    double u1 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0); // avoid log(0)
    double u2 = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double monte_carlo_pricer(option_spread option)
{
    double z = gaussian_random();
    
}

int main()
{
    int count = 0;
    option_spread *options = read_csv("Data/nvda_data_filtered.csv", &count);

    // Print entries read from file
    for (int i = 0; i < count; i++) 
    {
        printf("Entry %d:\n"
           "  Underlying: %.2f, Strike: %.2f, DTE: %.2f\n"
           "  Call - IV: %.6f, Mid: %.6f\n"
           "  Put  - IV: %.6f, Mid: %.6f\n"
           "  Risk-Free Rate: %.6f\n\n",
           i,
           options[i].underlying, options[i].strike, options[i].dte,
           options[i].c_iv, options[i].c_mid,
           options[i].p_iv, options[i].p_mid,
           options[i].rfr);
    }   

    return(0);
}