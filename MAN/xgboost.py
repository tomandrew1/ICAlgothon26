
import pandas as pd
import numpy as np
import xgboost as xgb

# --- Configuration ---
team_name = "RATT"
round_number = 1
output_filename = f"MAN/{team_name}_round_{round_number}.csv"

# 1. Load the Data
prices = pd.read_csv('MAN/prices.csv')
signals = pd.read_csv('MAN/signals.csv')
volumes = pd.read_csv('MAN/volumes.csv')
cash = pd.read_csv('MAN/cash_rate.csv')

# Convert date columns to standard datetime objects
for df in [prices, signals, volumes, cash]:
    df['date'] = pd.to_datetime(df['date'])

# 2. Merge and Restructure into "Panel" Data
print("Restructuring data into panel format...")
instruments = [f"INSTRUMENT_{i}" for i in range(1, 11)]
panel_data = []

for inst in instruments:
    # Create a clean dataframe for a single instrument
    df = prices[['date']].copy()
    df['asset'] = inst
    
    # Add Features (Prices, Volumes, and the 1-month risk-free rate)
    df['price'] = prices[inst]
    df['volume'] = volumes[f"{inst}_vol"]
    
    # We can map the cash rate using the date
    df = df.merge(cash[['date', '1mo']], on='date', how='left')
    df.rename(columns={'1mo': 'risk_free_1mo'}, inplace=True)
    
    # Add the trend signals
    for t in [4, 8, 16, 32]:
        df[f'trend{t}'] = signals[f"{inst}_trend{t}"]
    
    # Create the TARGET VARIABLE (What we are trying to predict)
    # We want to predict tomorrow's return: (Price_tomorrow - Price_today) / Price_today
    df['target_return'] = df['price'].shift(-1) / df['price'] - 1
    
    panel_data.append(df)

# Combine all 10 instruments into one massive dataset
panel = pd.concat(panel_data, ignore_index=True)

# Sort strictly by date so we don't accidentally leak future data into the past
panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)

# 3. Train / Predict Split
latest_date = panel['date'].max()
print(f"Latest available date for prediction: {latest_date.date()}")

# Define our feature columns
features = ['price', 'volume', 'risk_free_1mo', 'trend4', 'trend8', 'trend16', 'trend32']

# Training Data (Everything before the very last day)
# Drop any rows where features or targets are NaN (like the first few days of moving averages)
train_data = panel[panel['date'] < latest_date].dropna(subset=features + ['target_return'])
X_train = train_data[features]
y_train = train_data['target_return']

# Prediction Data (The very last day where we don't know the future return yet)
predict_data = panel[panel['date'] == latest_date].copy()
X_predict = predict_data[features]

# 4. Train the XGBoost Model
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(
    n_estimators=150,       # Number of decision trees
    learning_rate=0.05,     # Step size shrinkage
    max_depth=4,            # Depth of trees (keep low to prevent overfitting finance data)
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

# 5. Predict Returns for the Final Day
predict_data['predicted_return'] = model.predict(X_predict)

# 6. Allocate Portfolio Weights (Max Profit Strategy)
print("Allocating portfolio weights...")
predictions = predict_data['predicted_return'].values

# Rule 1: Only invest in assets we predict will go UP (ignore negatives)
positive_predictions = np.maximum(predictions, 0)

# Rule 2: Divide each prediction by the total sum so weights equal exactly 1.0 (100%)
if np.sum(positive_predictions) > 0:
    weights = positive_predictions / np.sum(positive_predictions)
else:
    # Fallback safety net: If the model predicts a market crash (all assets go down),
    # hold an equal 10% weight in everything (or alter this to hold 0% / go to cash)
    weights = np.ones(10) / 10.0

predict_data['weight'] = np.round(weights, 4) # Round to 4 decimal places

# 7. Format and Export Output
output = predict_data[['asset', 'weight']].copy()

# Fix rounding errors to ensure exact 1.0 sum (e.g., 0.9999 -> 1.0000)
diff = 1.0 - output['weight'].sum()
output.iloc[0, output.columns.get_loc('weight')] += diff

# Save to CSV
output.to_csv(output_filename, index=False)
print(f"\nSuccess! Saved to {output_filename}")
print("-" * 30)
print(output)
