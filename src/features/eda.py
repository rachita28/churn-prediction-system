"""
src/features/eda.py — Automated Visualizations for Real IBM Telco Dataset
"""
import os
import sys
import warnings
from pathlib import Path

# Path indexing verification
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_RAW, REPORTS_DIR, TARGET_COL

# Dark theme presentation configurations
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "#333",
    "grid.color": "#333",
})
ACCENT = "#00d4aa"
DANGER = "#ff4b6e"


def generate_eda_report(input_data="customer_data.csv"):
    """Accepts either a string filename or a pre-loaded pandas DataFrame"""
    print("Automated EDA plots generating...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple check: Agar dataframe pass hua hai toh direct use karein, nahi toh load karein
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        path = Path("data/raw") / str(input_data)
        if not path.exists():
            print(f"❌ Cannot find file for plotting: {path}")
            return
        df = pd.read_csv(path)
    
    # Plot 1: Target Churn Distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET_COL].value_counts()
    bars = ax.bar(["Retained (0)", "Churned (1)"], counts.values, color=[ACCENT, DANGER], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{yval:,}", ha='center', va='bottom', fontweight='bold')
    ax.set_title("Overall Customer Class Churn Balance", pad=15)
    plt.savefig(REPORTS_DIR / "churn_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Plot 2: Contract vs Churn Value
    fig, ax = plt.subplots(figsize=(8, 5))
    if "Contract" in df.columns:
        sns.countplot(data=df, x="Contract", hue=TARGET_COL, palette=[ACCENT, DANGER], ax=ax)
        ax.set_title("Churn Incidents Mapped Across Contract Type Groupings", pad=15)
        ax.set_ylabel("Account Volume")
        ax.set_xlabel("Contract Type")
        ax.legend(["Retained", "Churned"])
        plt.savefig(REPORTS_DIR / "contract_vs_churn.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("✅ Automated plotting graphs compiled successfully.")


def generate_text_summary(df: pd.DataFrame):
    """Safely calculates dataset insights matching the real IBM schema names"""
    total = len(df)
    churn_rate = df[TARGET_COL].mean() if TARGET_COL in df.columns else 0.0
    
    contract_col = "Contract" if "Contract" in df.columns else ("contract_type" if "contract_type" in df.columns else None)
    charges_col = "Monthly Charges" if "Monthly Charges" in df.columns else ("monthly_charges" if "monthly_charges" in df.columns else None)
    
    top_contract = "Unknown"
    avg_billing_churners = 0.0
    
    if contract_col and TARGET_COL in df.columns:
        top_contract = df.groupby(contract_col)[TARGET_COL].mean().idxmax()
        
    if charges_col and TARGET_COL in df.columns:
        avg_billing_churners = df[df[TARGET_COL] == 1][charges_col].mean()

    summary_text = f"""=======================================================
 📊 EXPLORATORY DATA ANALYSIS BUSINESS MATRIX REPORT
=======================================================
 🏢 Total Monitored Portfolios Account Base : {total:,} rows
 📉 Observed System Churn Probability Vector: {churn_rate:.1%}
 🚨 Highest Risk Contract Structures Channel: {top_contract}
 💸 Average Invoice Exposure (At-Risk User): ${avg_billing_churners:.2f}/mo
======================================================="""
    print(summary_text)


if __name__ == "__main__":
    raw_path = Path("data/raw/customer_data.csv")
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        generate_eda_report(raw_df)
        generate_text_summary(raw_df)