import os
import numpy as np
import pandas as pd

# Define input and output directories
input_dir = "OptionsDX_Raw_Data"
output_dir = "Data"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define column names (matching provided format)
columns = [
    "QUOTE_UNIXTIME", "QUOTE_READTIME", "QUOTE_DATE", "QUOTE_TIME_HOURS",
    "UNDERLYING_LAST", "EXPIRE_DATE", "EXPIRE_UNIX", "DTE",
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV",
    "C_VOLUME", "C_LAST", "C_SIZE", "C_BID", "C_ASK",
    "STRIKE", "P_BID", "P_ASK", "P_SIZE", "P_LAST",
    "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV",
    "P_VOLUME", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"
]

# Columns to convert to float
float_cols = [
    "UNDERLYING_LAST", "DTE", "C_BID", "C_ASK", "P_BID", "P_ASK", "STRIKE"
]

# Process each asset separately
for asset in os.listdir(input_dir):
    asset_path = os.path.join(input_dir, asset)
    if not os.path.isdir(asset_path):
        continue

    asset_data = []  # Store all data for this asset

    for file in os.listdir(asset_path):
        file_path = os.path.join(asset_path, file)
        if not file.endswith(".txt"):
            continue

        # Read CSV
        df = pd.read_csv(file_path, header=None, low_memory=False, dtype=str)

        # Ensure correct column count
        if df.shape[1] != len(columns):
            print(f"Skipping {file}: column mismatch ({df.shape[1]} found, {len(columns)} expected)")
            continue

        df.columns = columns  # Assign corrected column names

        # Convert relevant columns to float
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute mid-prices
        df["C_MID_PRICE"] = (df["C_BID"] + df["C_ASK"]) / 2
        df["P_MID_PRICE"] = (df["P_BID"] + df["P_ASK"]) / 2

        # Convert DTE from days to years
        df["T"] = df["DTE"] / 365.0

        # Compute risk-free interest rate (r) using options parity
        def compute_r(row):
            C, P, S, K, T = row["C_MID_PRICE"], row["P_MID_PRICE"], row["UNDERLYING_LAST"], row["STRIKE"], row["T"]
            if pd.isna(C) or pd.isna(P) or pd.isna(S) or pd.isna(K) or T <= 0:
                return np.nan  # Skip invalid rows

            try:
                ratio = (K / S) * (1 - ((C - P) / S))
                if ratio <= 0:  # Avoid log(0) or log(negative)
                    return np.nan
                return -np.log(ratio) / T
            except:
                return np.nan

        df["RISK_FREE_RATE"] = df.apply(compute_r, axis=1)

        # Keep relevant columns
        df = df[[
            "QUOTE_UNIXTIME", "UNDERLYING_LAST", "EXPIRE_UNIX", "DTE", "STRIKE",
            "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV", "C_VOLUME", "C_MID_PRICE",
            "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV", "P_VOLUME", "P_MID_PRICE",
            "RISK_FREE_RATE"
        ]]

        # Append processed data
        asset_data.append(df)

    # Save aggregated file **per asset**
    if asset_data:
        result_df = pd.concat(asset_data, ignore_index=True)
        
        # Sort by timestamp to ensure chronological order
        result_df = result_df.sort_values(by="QUOTE_UNIXTIME").reset_index(drop=True)

        # Save per asset
        output_file = os.path.join(output_dir, f"{asset}_data.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

print("Processing complete.")
