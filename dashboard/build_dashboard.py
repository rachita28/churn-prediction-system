"""
dashboard/build_dashboard.py — Uploaded Data Metrics Compiler
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from config import DATA_PROCESSED, MODELS_DIR

def compile_dashboard():
    print("🔮 Compiling Uploaded Churn Model Inference Outputs...")
    
    model_data = joblib.load(MODELS_DIR / "best_model.pkl")
    model = model_data["model"]
    
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    cleaned = pd.read_csv(DATA_PROCESSED / "cleaned_data.csv")
    
    probs = model.predict_proba(X_test)[:, 1]
    
    test_idx = y_test.index
    final_df = cleaned.iloc[test_idx].reset_index(drop=True)
    final_df["probability"] = probs
    
    top_risk = final_df.sort_values(by="probability", ascending=False).head(15)
    
    total_customers = len(final_df)
    critical_cases = len(final_df[final_df["probability"] >= 0.70])
    avg_charges = final_df["Monthly Charges"].mean()
    total_risk_revenue = final_df[final_df["probability"] >= 0.70]["Monthly Charges"].sum()
    
    customer_list = []
    for _, row in top_risk.iterrows():
        customer_list.append({
            "id": str(row["CustomerID"]),
            "contract": str(row["Contract"]),
            "tenure": int(row["Tenure Months"]),
            "charges": float(row["Monthly Charges"]),
            "prob": float(row["probability"] * 100)
        })
        
    customer_json = json.dumps(customer_list)

    html_layout = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enterprise Retention Control Center</title>
    <style>
        body {{ font-family: 'Space Grotesk', -apple-system, sans-serif; background: #080b14; color: #e2e8f0; margin: 0; padding: 25px; }}
        .navbar {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e2840; padding-bottom: 20px; margin-bottom: 25px; }}
        .currency-selector {{ background: #141928; color: #00d4aa; border: 1px solid #1e2840; padding: 8px 16px; border-radius: 8px; font-weight: bold; cursor: pointer; outline: none; }}
        .grid {{ display: flex; gap: 20px; margin-bottom: 25px; }}
        .card {{ background: #141928; border: 1px solid #1e2840; border-radius: 12px; padding: 20px; flex: 1; border-top: 4px solid #00d4aa; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .card.danger {{ border-top-color: #ff4b6e; }}
        .card-title {{ font-size: 11px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; }}
        .card-value {{ font-size: 30px; font-weight: bold; margin-top: 10px; font-family: monospace; }}
        .table-card {{ background: #141928; border: 1px solid #1e2840; border-radius: 12px; padding: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #1e2840; font-size: 14px; }}
        th {{ color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
        .badge {{ background: rgba(255,75,110,0.15); color: #ff4b6e; padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: bold; border: 1px solid rgba(255,75,110,0.3); }}
    </style>
</head>
<body>

    <div class="navbar">
        <h2 style="margin:0; font-weight:700;">📉 ChurnIQ — Uploaded Dataset Control Room Panel</h2>
        <div>
            <label style="color: #64748b; font-size: 13px; margin-right: 8px; font-weight:500;">Select Account Display Currency Group:</label>
            <select class="currency-selector" id="currencySelect" onchange="updateCurrency()">
                <option value="INR">INR (₹) — Indian Rupee Ledger</option>
                <option value="USD">USD ($) — Global Base Account</option>
                <option value="EUR">EUR (€) — Euro Zone Portfolio</option>
                <option value="GBP">GBP (£) — United Kingdom Market</option>
            </select>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Analyzed Records Matrix Split Length</div>
            <div class="card-value" style="color: #818cf8;">{total_customers:,}</div>
        </div>
        <div class="card danger">
            <div class="card-title">ML-Flagged Critical Outliers</div>
            <div class="card-value" style="color: #ff4b6e;">{critical_cases}</div>
        </div>
        <div class="card danger">
            <div class="card-title">Calculated At-Risk Monthly Revenue</div>
            <div class="card-value" id="revAtRisk" style="color: #ff8c42;"></div>
        </div>
        <div class="card">
            <div class="card-title">Average Observed Invoiced Charging</div>
            <div class="card-value" id="avgBilling" style="color: #00d4aa;"></div>
        </div>
    </div>

    <div class="table-card">
        <h3 style="margin:0 0 10px 0;">🚨 Live Operational Risk Matrix (Top 15 Most Fragile Customer Portfolios)</h3>
        <table>
            <thead>
                <tr>
                    <th>Customer Serial ID</th>
                    <th>Structural Contract Type</th>
                    <th>Observed Account Tenure</th>
                    <th>Calculated Monthly Charge</th>
                    <th>Calibrated Predictive Vector Score</th>
                    <th>Corporate Strategic Outreach</th>
                </tr>
            </thead>
            <tbody id="customerTableBody"></tbody>
        </table>
    </div>

    <script>
        const rates = {{ "INR": 83.50, "USD": 1.00, "EUR": 0.92, "GBP": 0.79 }};
        const symbols = {{ "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£" }};
        
        const baseRevAtRisk = {total_risk_revenue};
        const baseAvgBilling = {avg_charges};
        const rawCustomers = {customer_json};

        function updateCurrency() {{
            const curr = document.getElementById("currencySelect").value;
            const rate = rates[curr];
            const sym = symbols[curr];

            document.getElementById("revAtRisk").innerText = sym + (baseRevAtRisk * rate).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
            document.getElementById("avgBilling").innerText = sym + (baseAvgBilling * rate).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});

            const tbody = document.getElementById("customerTableBody");
            tbody.innerHTML = "";
            
            rawCustomers.forEach(c => {{
                let convertedCharges = (c.charges * rate).toFixed(2);
                tbody.innerHTML += `
                    <tr>
                        <td style="color:#818cf8; font-family:monospace; font-weight:600;">${{c.id}}</td>
                        <td>${{c.contract}}</td>
                        <td>${{c.tenure}} Months</td>
                        <td style="font-family:monospace; color:#00d4aa; font-weight:600;">${{sym}}${{parseFloat(convertedCharges).toLocaleString(undefined, {{minimumFractionDigits: 2}})}}</td>
                        <td style="font-weight:bold; color:#ff8c42; font-family:monospace;">${{c.prob.toFixed(1)}}%</td>
                        <td><span class="badge">Immediate Outreach</span></td>
                    </tr>
                `;
            }});
        }}

        updateCurrency();
    </script>
</body>
</html>
    """
    
    output_path = Path("dashboard/churniq_dashboard.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_layout)
    print(f"🎉 Fully Connected Multi-Currency Executive Dashboard written to: {output_path}")

if __name__ == "__main__":
    compile_dashboard()