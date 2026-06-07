"""
src/data/preprocess.py — Robust Processing Pipeline for IBM Telco Data
"""
import os
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CATEGORICAL_COLS,
    DATA_PROCESSED,
    DATA_RAW,
    ID_COL,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

def load_raw(filename: str = "customer_data.csv") -> pd.DataFrame:
    path = DATA_RAW / filename
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df):,} records from dataset.")
    return df

def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET_COL] + NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Structural mismatch! Missing columns: {missing}")
    print("✅ Schema verification passed.")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(subset=ID_COL, keep="first", inplace=True)
    
    # Handle blank spaces inside Total Charges cleanly before scaling
    if "Total Charges" in df.columns:
        df["Total Charges"] = df["Total Charges"].astype(str).str.strip().replace(r'^\s*$', np.nan, regex=True)
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        # Null values ko Monthly Charges se fill karenge
        df["Total Charges"].fillna(df["Monthly Charges"], inplace=True)
        
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().replace(r'^\s*$', "Unknown", regex=True)
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_revenue"] = df["Total Charges"] / df["Tenure Months"].replace(0, 1)
    df["contract_risk"] = df["Contract"].map({"Month-to-month": 3, "One year": 1, "Two year": 0}).fillna(1)
    print(f"⚙️  Engineered tracking metrics successfully.")
    return df

def encode_and_scale(df: pd.DataFrame):
    df = df.copy()
    drop_cols = [ID_COL, TARGET_COL]
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    
    # Select features explicitly by type to avoid any tracking leakage
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns] + ["Contract"]
    cat_cols = list(set(cat_cols)) # Duplicates remove karne ke liye
    
    le_dict = {}
    for col in cat_cols:
        if col in feature_df.columns:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            le_dict[col] = le
            
    # Kisi bhi extra non-numeric text columns ko safe drop karenge
    feature_df = feature_df.select_dtypes(include=[np.number])
    
    # Check for any residual NaNs and fill them to prevent Logistic Regression crash
    if feature_df.isna().sum().sum() > 0:
        feature_df.fillna(0, inplace=True)
        
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    feature_df_scaled = pd.DataFrame(scaled_features, columns=feature_df.columns)
    
    return feature_df_scaled, df[TARGET_COL].astype(int), scaler, le_dict

def save_splits(X, y, scaler, le_dict):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
    
    joblib.dump(scaler, DATA_PROCESSED / "scaler.pkl")
    joblib.dump(le_dict, DATA_PROCESSED / "label_encoders.pkl")
    print(f"💾 Train/Test metrics matrices securely saved.")
    return X_train, X_test, y_train, y_test

def run_pipeline(filename: str = "customer_data.csv"):
    df = load_raw(filename)
    df = validate(df)
    df = clean(df)
    df = engineer_features(df)
    X, y, scaler, le_dict = encode_and_scale(df)
    save_splits(X, y, scaler, le_dict)
    print("✅ Preprocessing pipeline complete!")

if __name__ == "__main__":
    run_pipeline()