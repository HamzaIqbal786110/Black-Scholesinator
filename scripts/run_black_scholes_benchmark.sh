#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=black_scholes_bench
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --cpus-per-task=10
#SBATCH --output=scripts/outputs/master_benchmark.out

# === Change to your project root ===
cd "$HOME/Black-Scholesinator"

# Create output directory if it doesn't exist
mkdir -p outputs

# Programs to test
PROGRAMS=(
    "bin/black_scholes_base"
    "bin/black_scholes_cpu_acc"
    "bin/black_scholes_gpu"
    "bin/black_scholes_hybrid1"
    "bin/black_scholes_hybrid2"
)

# Summary timing file
TIMING_FILE="outputs/summary.txt"
> "$TIMING_FILE"

echo "Starting Benchmarks..." | tee -a "$TIMING_FILE"
echo "=======================" >> "$TIMING_FILE"

# Loop over all combinations of psteps and tsteps
for psteps in {20..200..20}; do
    for tsteps in {10000..100000..10000}; do

        echo "Configuration: p_steps=$psteps, t_steps=$tsteps" | tee -a "$TIMING_FILE"

        for prog in "${PROGRAMS[@]}"; do
            PROG_NAME=$(basename "$prog")
            OUTFILE="outputs/${PROG_NAME}_p${psteps}_t${tsteps}.out"

            if [[ ! -f "$prog" ]]; then
                echo "Binary not found: $prog" | tee -a "$TIMING_FILE"
                continue
            fi

            echo "Benchmarking: $PROG_NAME with p=$psteps, t=$tsteps"
            echo "Benchmarking: $PROG_NAME with p=$psteps, t=$tsteps" > "$OUTFILE"
            total_time=0
            num_runs=10  # Change to 100 when ready

            for run in $(seq 1 $num_runs); do
                echo "Run $run" >> "$OUTFILE"
                run_time=$($prog $psteps $tsteps | tee -a "$OUTFILE" | grep "TIME TAKEN" | awk '{print $NF}')
                if [[ -z "$run_time" ]]; then
                    echo "Run $run failed or TIME TAKEN not found" >> "$OUTFILE"
                else
                    total_time=$(echo "$total_time + $run_time" | bc)
                fi
            done

            if [[ "$total_time" == 0 ]]; then
                echo "$PROG_NAME failed for p=$psteps t=$tsteps" >> "$TIMING_FILE"
            else
                avg_time=$(echo "scale=5; $total_time / $num_runs" | bc)
                echo "$PROG_NAME average time for p=$psteps, t=$tsteps: $avg_time ms" >> "$TIMING_FILE"
            fi

            echo "" >> "$TIMING_FILE"
        done
    done
done

echo "All benchmarks complete." >> "$TIMING_FILE"
