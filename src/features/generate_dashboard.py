import os
import joblib
import numpy as np
import pandas as pd


def build_dashboard(test_path, model_path, output_html_path):
    print("Loading model and calculating risk segments...")
    # Load the test dataset and our trained model pipeline
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)

    # Extract target and features
    X_test = test_df.drop(columns=["Churn"])
    y_true = test_df["Churn"]

    # Predict the mathematical probability of churning (0.0 to 1.0)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Build a clean reporting dataframe
    report_df = pd.DataFrame(
        {
            "MonthlyCharges": X_test["MonthlyCharges"],
            "TenureMonths": X_test["TenureMonths"],
            "Contract": X_test["Contract"],
            "InternetService": X_test["InternetService"],
            "ActualChurn": y_true,
            "ChurnProbability": probabilities,
        }
    )

    # Segment customers into risk tiers based on their probability score
    conditions = [
        (report_df["ChurnProbability"] >= 0.7),
        (report_df["ChurnProbability"] >= 0.3)
        & (report_df["ChurnProbability"] < 0.7),
        (report_df["ChurnProbability"] < 0.3),
    ]
    choices = ["High Risk", "Medium Risk", "Low Risk"]
    report_df["RiskSegment"] = np.select(conditions, choices, default="Low Risk")

    # Calculate high-level business summary metrics
    total_customers = len(report_df)
    high_risk_count = len(report_df[report_df["RiskSegment"] == "High Risk"])
    avg_churn_prob = report_df["ChurnProbability"].mean() * 100

    # Calculate revenue currently at risk (sum of monthly charges for high risk accounts)
    revenue_at_risk = report_df[report_df["RiskSegment"] == "High Risk"][
        "MonthlyCharges"
    ].sum()

    print("Generating Interactive HTML Executive Dashboard...")

    # Build a clean, modern HTML layout string with CSS styling
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Executive Churn Analytics Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
        .header {{ background: #1e293b; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics-container {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #3b82f6; }}
        .card.danger {{ border-top-color: #ef4444; }}
        .card h3 {{ margin: 0; color: #64748b; font-size: 14px; text-transform: uppercase; }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #0f172a; margin-top: 10px; }}
        .table-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background-color: #f8fafc; color: #475569; }}
        .badge {{ padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .badge.high {{ background: #fee2e2; color: #991b1b; }}
        .badge.med {{ background: #fef3c7; color: #92400e; }}
        .badge.low {{ background: #dcfce7; color: #166534; }}
    </style>
</head>
<body>

    <div class="header">
        <h1 style="margin:0;">Telecom Customer Retention Command Center</h1>
        <p style="margin:5px 0 0 0; opacity: 0.8;">Real-time risk assessment and revenue preservation insights</p>
    </div>

    <div class="metrics-container">
        <div class="card">
            <h3>Total Evaluated Accounts</h3>
            <div class="value">{total_customers:,}</div>
        </div>
        <div class="card danger">
            <h3>High-Risk Customers</h3>
            <div class="value">{high_risk_count:,}</div>
        </div>
        <div class="card danger">
            <h3>Monthly Revenue at Risk</h3>
            <div class="value">${revenue_at_risk:,.2f}</div>
        </div>
        <div class="card">
            <h3>Overall Fleet Churn Rate</h3>
            <div class="value">{avg_churn_prob:.1f}%</div>
        </div>
    </div>

    <div class="table-container">
        <h2>At-Risk Customer Action Registry (Top 15 Most Critical Accounts)</h2>
        <table>
            <thead>
                <tr>
                    <th>Contract Type</th>
                    <th>Internet Infrastructure</th>
                    <th>Monthly Bill</th>
                    <th>Account Tenure</th>
                    <th>Calculated Churn Risk</th>
                    <th>Priority Action Segment</th>
                </tr>
            </thead>
            <tbody>
    """

    # Populate table with the highest risk accounts
    top_at_risk = report_df.sort_values(
        by="ChurnProbability", ascending=False
    ).head(15)
    for _, row in top_at_risk.iterrows():
        badge_class = (
            "high"
            if row["RiskSegment"] == "High Risk"
            else ("med" if row["RiskSegment"] == "Medium Risk" else "low")
        )
        html_content += f"""
                <tr>
                    <td>{row['Contract']}</td>
                    <td>{row['InternetService']}</td>
                    <td>${row['MonthlyCharges']:.2f}</td>
                    <td>{row['TenureMonths']} mos</td>
                    <td>{(row['ChurnProbability']*100):.1f}%</td>
                    <td><span class="badge {badge_class}">{row['RiskSegment']}</span></td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </div>

</body>
</html>
    """

    # Save the generated interactive dashboard HTML file
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Dashboard successfully compiled and exported to: {output_html_path}")


if __name__ == "__main__":
    build_dashboard(
        test_path="data/processed/test_cleaned.csv",
        model_path="src/models/xgboost_churn_model.pkl",
        output_html_path="dashboard/executive_dashboard.html",
    )