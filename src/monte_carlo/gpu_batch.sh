#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=hpcfinal_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu


OUTPUT_FILE="monte_carlo_gpu.out"

> $OUTPUT_FILE

./hpc/hw5/mcgpu >> $OUTPUT_FILE
