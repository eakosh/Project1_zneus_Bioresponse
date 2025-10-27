import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pickle
import os
from datetime import datetime


class Config:
    DATA_PATH = "../data/phpSSK7iA.csv"
    OUTPUT_DIR = "../processed_data"
    REMOVE_DUPLICATES = True
    REMOVE_ZERO_COLUMNS = True
    REMOVE_CONSTANT_COLUMNS = True
    VARIANCE_THRESHOLD = 0.01
    TEST_SIZE = 0.20
    VAL_SIZE = 0.20
    RANDOM_STATE = 42
    STRATIFY = True
    NORMALIZATION_METHOD = 'standard'
    APPLY_FEATURE_SELECTION = False
    TOP_N_FEATURES = 300


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting preprocessing...\n")

# 1. Load data
print("Loading dataset...")
try:
    df = pd.read_csv(Config.DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
except FileNotFoundError:
    print(f"ERROR: File not found at {Config.DATA_PATH}")
    exit(1)

if 'target' not in df.columns:
    print("ERROR: 'target' column not found.")
    exit(1)

# 2. Duplicates
if Config.REMOVE_DUPLICATES:
    n_dup = df.duplicated().sum()
    if n_dup:
        df = df.drop_duplicates()
        print(f"Removed {n_dup} duplicate rows.")
    else:
        print("No duplicate rows found.")

# 3. Missing values
missing = df.isnull().sum().sum()
if missing:
    print(f"Found {missing} missing values.")
    if missing < len(df) * 0.01:
        df = df.dropna()
        print("Dropped rows with missing values (less than 1%).")
    else:
        print("Warning: consider imputation strategy.")
else:
    print("No missing values found.")

# 4. Zero or constant columns
if Config.REMOVE_ZERO_COLUMNS:
    zero_cols = [c for c in df.columns if c != 'target' and (df[c] == 0).all()]
    if zero_cols:
        df = df.drop(columns=zero_cols)
        print(f"Removed {len(zero_cols)} zero-only columns.")
if Config.REMOVE_CONSTANT_COLUMNS:
    const_cols = [c for c in df.columns if c != 'target' and df[c].nunique() <= 1]
    if const_cols:
        df = df.drop(columns=const_cols)
        print(f"Removed {len(const_cols)} constant columns.")

# 5. Low variance
features = [c for c in df.columns if c != 'target']
X_temp, y = df[features], df['target']
variance = X_temp.var()
low_var = variance[variance < Config.VARIANCE_THRESHOLD].index.tolist()
if low_var:
    print(f"Removing {len(low_var)} low-variance features (<{Config.VARIANCE_THRESHOLD}).")
    selector = VarianceThreshold(Config.VARIANCE_THRESHOLD)
    X_temp = selector.fit_transform(X_temp)
    selected = [f for f, keep in zip(features, selector.get_support()) if keep]
    df = pd.concat([pd.DataFrame(X_temp, columns=selected, index=df.index), y], axis=1)
else:
    print("No low-variance features removed.")

# 6. Split
print("\nSplitting data...")
features = [c for c in df.columns if c != 'target']
X, y = df[features], df['target']
stratify = y if Config.STRATIFY else None

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                  random_state=Config.RANDOM_STATE, stratify=stratify)
val_size_adj = Config.VAL_SIZE / (1 - Config.TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adj,
                                                  random_state=Config.RANDOM_STATE, stratify=y_temp if Config.STRATIFY else None)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# 7. Normalization
print("\nNormalizing data...")
if Config.NORMALIZATION_METHOD == 'standard':
    scaler = StandardScaler()
elif Config.NORMALIZATION_METHOD == 'minmax':
    scaler = MinMaxScaler()
else:
    print(f"ERROR: Unknown normalization method '{Config.NORMALIZATION_METHOD}'")
    exit(1)

scaler.fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=features)
X_val_s = pd.DataFrame(scaler.transform(X_val), columns=features)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features)
print("Normalization complete.")

# 8. Save
print("\nSaving processed files...")
X_train_s.to_csv(f"{Config.OUTPUT_DIR}/X_train.csv", index=False)
X_val_s.to_csv(f"{Config.OUTPUT_DIR}/X_val.csv", index=False)
X_test_s.to_csv(f"{Config.OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{Config.OUTPUT_DIR}/y_train.csv", index=False)
y_val.to_csv(f"{Config.OUTPUT_DIR}/y_val.csv", index=False)
y_test.to_csv(f"{Config.OUTPUT_DIR}/y_test.csv", index=False)
print("CSVs saved.")

with open(f"{Config.OUTPUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

with open(f"{Config.OUTPUT_DIR}/feature_names.txt", "w") as f:
    f.write("\n".join(features))

print(f"\nDone. Processed data saved to {Config.OUTPUT_DIR}/")
print(f"Total features: {len(features)}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")