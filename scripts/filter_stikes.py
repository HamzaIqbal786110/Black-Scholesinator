import csv

INPUT_CSV = "nvda_data.csv"
OUTPUT_CSV = "nvda_data_filtered.csv"
MAX_DEVIATION = 0.20  # 20%

def is_near_the_money(underlying, strike):
    if underlying == 0:
        return False
    lower_bound = underlying * (1 - MAX_DEVIATION)
    upper_bound = underlying * (1 + MAX_DEVIATION)
    return lower_bound <= strike <= upper_bound

with open(INPUT_CSV, newline='') as infile, open(OUTPUT_CSV, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    
    writer.writeheader()
    for row in reader:
        try:
            underlying = float(row["UNDERLYING_LAST"])
            strike = float(row["STRIKE"])
            if is_near_the_money(underlying, strike):
                writer.writerow(row)
        except ValueError:
            # Skip rows with bad/missing numeric data
            continue

print(f"Filtered data written to {OUTPUT_CSV}")
