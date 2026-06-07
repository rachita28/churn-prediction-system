"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic customer churn dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 7043  # Realistic dataset size

def generate():
    # --- Core demographics ---
    tenure          = np.random.randint(1, 73, N)
    age_group       = np.random.choice(["18-25","26-35","36-50","51+"], N, p=[0.15,0.30,0.35,0.20])
    region          = np.random.choice(["North","South","East","West"], N)

    # --- Product & service ---
    contract_type   = np.random.choice(["Month-to-month","One year","Two year"], N, p=[0.55,0.24,0.21])
    internet_svc    = np.random.choice(["DSL","Fiber optic","No"], N, p=[0.34,0.44,0.22])
    num_products    = np.random.randint(1, 6, N)
    payment_method  = np.random.choice(
        ["Electronic check","Mailed check","Bank transfer","Credit card"], N, p=[0.34,0.23,0.22,0.21]
    )

    # --- Service quality (correlated features) ---
    tech_support    = np.where(internet_svc=="No", "No internet service",
                        np.random.choice(["Yes","No"], N, p=[0.29,0.71]))
    online_security = np.where(internet_svc=="No", "No internet service",
                        np.random.choice(["Yes","No"], N, p=[0.28,0.72]))
    num_support_calls = np.random.poisson(1.5, N).clip(0, 10)
    satisfaction_score = np.random.choice([1,2,3,4,5], N, p=[0.10,0.15,0.25,0.30,0.20])

    # --- Financials ---
    base_charge     = np.where(internet_svc=="Fiber optic", 70, np.where(internet_svc=="DSL", 50, 20))
    monthly_charges = (base_charge + num_products * 5
                       + np.random.normal(0, 8, N)).clip(18, 120).round(2)
    total_charges   = (monthly_charges * tenure + np.random.normal(0, 50, N)).clip(0).round(2)

    # --- Churn probability (realistic business logic) ---
    churn_prob = (
        0.05                                                          # baseline
        + 0.25 * (contract_type == "Month-to-month")                 # risky contract
        + 0.10 * (payment_method == "Electronic check")              # friction signal
        + 0.15 * (internet_svc == "Fiber optic")                     # higher expectations
        + 0.08 * (num_support_calls > 3)                             # frustrated users
        - 0.10 * (satisfaction_score >= 4)                           # happy users stay
        - 0.12 * (tenure > 24)                                       # loyalty
        - 0.08 * (num_products >= 3)                                  # stickiness
        + 0.05 * (tech_support == "No")                              # no safety net
        + np.random.normal(0, 0.05, N)                               # noise
    ).clip(0, 0.95)

    churn = (np.random.rand(N) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id"       : [f"CUST-{i:05d}" for i in range(1, N+1)],
        "tenure"            : tenure,
        "monthly_charges"   : monthly_charges,
        "total_charges"     : total_charges,
        "contract_type"     : contract_type,
        "payment_method"    : payment_method,
        "internet_service"  : internet_svc,
        "tech_support"      : tech_support,
        "online_security"   : online_security,
        "num_products"      : num_products,
        "num_support_calls" : num_support_calls,
        "satisfaction_score": satisfaction_score,
        "age_group"         : age_group,
        "region"            : region,
        "churn"             : churn,
    })

    out = Path(__file__).parent / "raw" / "customer_data.csv"
    df.to_csv(out, index=False)
    print(f"✅ Dataset saved → {out}")
    print(f"   Rows: {len(df):,}  |  Churn rate: {churn.mean():.1%}")
    return df

if __name__ == "__main__":
    generate()