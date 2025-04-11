import os
import re
import matplotlib.pyplot as plt

# Directory where the output files are stored
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# File names
FILES = {
    "Basic": "black_scholes_base.out",
    "CPU Accelerated": "black_scholes_cpu_acc.out",
    "GPU Accelerated": "black_scholes_gpu.out"
}

def extract_times(filepath):
    times = []
    pattern = re.compile(r"TIME TAKEN\s*=\s*([\d.]+)\s*ms")
    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))
    return times

def main():
    avg_times = {}

    for label, filename in FILES.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
            continue

        times = extract_times(filepath)
        if times:
            avg_times[label] = sum(times) / len(times)
            print(f"{label} Average Time: {avg_times[label]:.2f} ms")
        else:
            print(f"No timing data found in {filename}")

    # Plotting
    if avg_times:
        labels = list(avg_times.keys())
        values = [avg_times[label] for label in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title("Average Execution Time of Black-Scholes Implementations")
        plt.ylabel("Time (ms)")
        plt.xlabel("Implementation")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_comparison.png"))
        plt.show()

if __name__ == "__main__":
    main()
