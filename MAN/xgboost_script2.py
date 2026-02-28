import pandas as pd
import numpy as np
import xgboost as xgb

# --- Configuration ---
team_name = "RATT"
round_number = 2
output_filename = f"MAN/{team_name}_round_{round_number}.csv"


# 1. Load the Data
try:
    prices = pd.read_csv('../man-imperial-algothon-2026/data/2025-02-28/prices.csv')
    signals = pd.read_csv('../man-imperial-algothon-2026/data/2025-02-28/signals.csv')
    volumes = pd.read_csv('../man-imperial-algothon-2026/data/2025-02-28/volumes.csv')
    cash = pd.read_csv('../man-imperial-algothon-2026/data/2025-02-28/cash_rate.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Ensure all CSVs are in the same directory.")
    exit()

# Convert dates to standard datetime format
for df in [prices, signals, volumes, cash]:
    df['date'] = pd.to_datetime(df['date'])

# Forward-fill any missing interest rate days (e.g., weekends/bank holidays)
cash = cash.sort_values('date').ffill()

print("Engineering stationary and macroeconomic features...")
instruments = [f"INSTRUMENT_{i}" for i in range(1, 11)]
panel_data = []

for inst in instruments:
    df = prices[['date']].copy()
    df['asset'] = inst
    
    price_series = prices[inst]
    vol_series = volumes[f"{inst}_vol"]
    
    # --- STATIONARY FEATURES ---
    df['return_1d'] = price_series.pct_change(1)
    df['return_5d'] = price_series.pct_change(5)
    df['return_20d'] = price_series.pct_change(20)
    
    vol_ma_20 = vol_series.rolling(window=20).mean()
    df['vol_ratio'] = vol_series / (vol_ma_20 + 1e-8)
    
    for t in [4, 8, 16, 32]:
        df[f'trend{t}'] = signals[f"{inst}_trend{t}"]
        
    # --- MACROECONOMIC FEATURES ---
    df = df.merge(cash[['date', '1mo']], on='date', how='left')
    df['daily_risk_free'] = (df['1mo'] / 100) / 252 
    df['rate_change'] = df['1mo'] - df['1mo'].shift(1)
    
    # --- TARGET VARIABLE (EXCESS RETURN) ---
    raw_target_return = price_series.shift(-1) / price_series - 1
    rf_tomorrow = df['daily_risk_free'].shift(-1)
    df['target_excess_return'] = raw_target_return - rf_tomorrow
    
    panel_data.append(df)

# Combine all instruments into our final panel dataset
panel = pd.concat(panel_data, ignore_index=True)
panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)

# 3. Train / Predict Split
latest_date = panel['date'].max()

features = [
    'return_1d', 'return_5d', 'return_20d', 'vol_ratio', 
    'trend4', 'trend8', 'trend16', 'trend32', 
    'daily_risk_free', 'rate_change'
]

# Training Data
train_data = panel[panel['date'] < latest_date].dropna(subset=features + ['target_excess_return'])
X_train = train_data[features]
y_train = train_data['target_excess_return']

# Prediction Data
predict_data = panel[panel['date'] == latest_date].copy()
X_predict = predict_data[features]

# 4. Train the XGBoost Model
print(f"Training XGBoost Regressor on data up to {latest_date.date()}...")
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

# 5. Predict Excess Returns for the Final Day
predict_data['predicted_excess_return'] = model.predict(X_predict)

# 6. Allocate Portfolio Weights (Strictly Non-Negative & Sums to 1)
print("Allocating portfolio weights...")
predictions = predict_data['predicted_excess_return'].values

# Enforce Non-Negative Rule
pos_preds = np.maximum(predictions, 0)

if np.sum(pos_preds) > 0:
    raw_weights = pos_preds / np.sum(pos_preds)
else:
    # Safe fallback if the model hates all assets today
    raw_weights = np.ones(len(predictions)) / len(predictions)

# --- The Largest Remainder Method (Guarantees perfect 1.0000 sum) ---
# Scale up to integers representing 1/10000ths
weights_scaled = raw_weights * 10000
weights_floor = np.floor(weights_scaled).astype(int)
remainders = weights_scaled - weights_floor

# Calculate how many 0.0001 "pennies" we are short of 1.0000 (10000)
missing_pennies = int(10000 - np.sum(weights_floor))

# Distribute the missing pennies to the assets whose fractions were closest to rounding up
if missing_pennies > 0:
    largest_remainder_indices = np.argsort(remainders)[-missing_pennies:]
    for idx in largest_remainder_indices:
        weights_floor[idx] += 1

# Convert back to decimals
final_weights = weights_floor / 10000.0
predict_data['weight'] = final_weights

# 7. Format and Export Output
output = predict_data[['asset', 'weight']].copy()

# Final safety checks before saving
assert np.isclose(output['weight'].sum(), 1.0), "CRITICAL ERROR: Weights do not sum to 1.0!"
assert (output['weight'] >= 0).all(), "CRITICAL ERROR: Negative weights detected!"

output.to_csv(output_filename, index=False)
print(f"\nSuccess! Saved to {output_filename}")
print(f"Total Weight Sum: {output['weight'].sum():.4f}")
print("-" * 30)
print(output)