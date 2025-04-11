#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=black_scholes_bench
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --cpus-per-task=10
#SBATCH --output=scripts/outputs/master_benchmark.out

# === FIX: cd into project root ===
cd "$HOME/Black-Scholesinator"  # or wherever your project root is

# Create output directory if it doesn't exist
mkdir -p outputs

# Paths (relative to project root)
OUTDIR="outputs"
TIMING_FILE="$OUTDIR/summary.txt"
PROGRAMS=("bin/black_scholes_base" "bin/black_scholes_cpu_acc" "bin/black_scholes_gpu")

# Clean timing file
> "$TIMING_FILE"

echo "Running benchmarks..."
echo "=====================" >> "$TIMING_FILE"

for prog in "${PROGRAMS[@]}"; do
    BIN="$prog"
    PROG_NAME=$(basename "$prog")
    OUTFILE="$OUTDIR/${PROG_NAME}.out"

    if [[ ! -f "$BIN" ]]; then
        echo "Required binary $BIN not found. Exiting." | tee -a "$TIMING_FILE"
        exit 1
    fi

    echo "Benchmarking: $prog"
    echo "Benchmarking: $prog" >> "$OUTFILE"
    total_time=0

    for run in {1..100}; do   # 👈 set to 1 while testing
        echo "Run $run" >> "$OUTFILE"
        run_time=$($BIN | tee -a "$OUTFILE" | grep "TIME TAKEN" | awk '{print $NF}')
        if [[ -z $run_time ]]; then
            echo "Run $run failed or timing not found" >> "$OUTFILE"
        else
            total_time=$(echo "$total_time + $run_time" | bc)
        fi
    done

    if [[ "$total_time" == 0 ]]; then
        echo "$prog average time: FAILED" >> "$TIMING_FILE"
    else
        avg_time=$(echo "scale=5; $total_time / 1" | bc)
        echo "$prog average time: $avg_time ms" >> "$TIMING_FILE"
    fi

    echo "" >> "$TIMING_FILE"
done
