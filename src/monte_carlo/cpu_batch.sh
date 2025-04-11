#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=hpcfinal_gpu
#SBATCH --partition9=courses
#SBATCH --constraint=broadwell


OUTPUT_FILE="monte_carlo_cpu.out"

> $OUTPUT_FILE

./hpc/hw5/mcgpu >> $OUTPUT_FILE
