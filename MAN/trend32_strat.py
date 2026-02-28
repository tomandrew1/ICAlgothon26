import pandas as pd
import numpy as np

# --- Configuration ---
team_name = "RATT"
round_number = 1
output_filename = f"MAN/{team_name}_round_{round_number}.csv"

# 1. Load the signals data
signals_df = pd.read_csv('MAN/signals.csv')

# 2. Get the most recent day's data (the last row)
latest_signals = signals_df.iloc[-1]
latest_date = latest_signals['date']
print(f"Generating portfolio for date: {latest_date}")

# 3. Extract the trend32 signals for all 10 instruments
instruments = [f"INSTRUMENT_{i}" for i in range(1, 11)]
trend_values = []

for inst in instruments:
    column_name = f"{inst}_trend32"
    # Get the signal value, default to 0 if it's missing (NaN)
    signal_value = latest_signals.get(column_name, 0)
    if pd.isna(signal_value):
        signal_value = 0
    trend_values.append(signal_value)

# Convert to a numpy array for easy math
trend_values = np.array(trend_values)

# 4. Allocate Weights (Only buy positive trends, proportional to strength)
# Replace any negative trends with 0
positive_trends = np.maximum(trend_values, 0)

# Calculate weights so they sum to 1.0
total_positive_trend = np.sum(positive_trends)

if total_positive_trend > 0:
    weights = positive_trends / total_positive_trend
else:
    # Fallback: If ALL trends are negative, hold equal weight in everything
    # (Or you could hold cash, depending on the competition rules!)
    weights = np.ones(10) / 10.0

# Round to 4 decimal places for clean output
weights = np.round(weights, 4)

# 5. Format and Export to CSV
output_data = pd.DataFrame({
    'asset': instruments,
    'weight': weights
})

output_data.to_csv(output_filename, index=False)
print(f"Success! Portfolio weights saved to {output_filename}")

# Display the final allocation
print("\nFinal Portfolio Allocation:")
print(output_data)
