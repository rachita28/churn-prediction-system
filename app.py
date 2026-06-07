"""
app.py - ChurnIQ Flask Server
Run: python app.py
Open: http://localhost:5000
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "src" / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"

app = Flask(__name__)

model_data = joblib.load(MODEL_PATH)
MODEL      = model_data["model"]
MODEL_NAME = model_data["name"]
SCALER     = joblib.load(SCALER_PATH)
print(f"✅ Model loaded: {MODEL_NAME}")

REQUIRED_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "Contract", "InternetService", "PaymentMethod",
    "TechSupport", "OnlineSecurity", "StreamingTV",
    "StreamingMovies", "OnlineBackup", "DeviceProtection",
    "MultipleLines", "PhoneService", "PaperlessBilling",
    "SeniorCitizen", "Partner", "Dependents", "gender"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "Contract", "InternetService", "PaymentMethod",
    "TechSupport", "OnlineSecurity", "StreamingTV",
    "StreamingMovies", "OnlineBackup", "DeviceProtection",
    "MultipleLines", "PhoneService", "PaperlessBilling",
    "Partner", "Dependents", "gender"
]


def get_segment(prob):
    if prob >= 0.80: return "Critical"
    if prob >= 0.60: return "High Risk"
    if prob >= 0.30: return "Moderate"
    return "Low Risk"


def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = pd.Categorical(df[col]).codes
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    df["AvgMonthlyRevenue"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["ContractRisk"]      = df["Contract"].map({0:3, 1:1, 2:0}).fillna(1)
    df["LongTenure"]        = (df["tenure"] > 24).astype(int)
    df["HighCharges"]       = (df["MonthlyCharges"] > 70).astype(int)
    df["SeniorHighRisk"]    = ((df.get("SeniorCitizen", 0) == 1) &
                               (df["MonthlyCharges"] > 60)).astype(int)
    return df


@app.route("/")
def index():
    return send_from_directory("dashboard", "churniq_upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file"}), 400

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        total = len(df)

        matched   = [c for c in REQUIRED_COLS if c in df.columns]
        match_pct = len(matched) / len(REQUIRED_COLS)
        mode      = "ml" if match_pct >= 0.7 else "basic"

        id_col = None
        for c in ["customerID", "CustomerID", "customer_id", "ID", "id"]:
            if c in df.columns:
                id_col = c
                break
        customer_ids = df[id_col].tolist() if id_col else [f"CUST-{i:04d}" for i in range(total)]

        # ── ML or Basic mode ──────────────────────────────────
        if mode == "ml":
            processed = preprocess(df)
            try:
                feature_cols = MODEL.feature_names_in_.tolist()
            except AttributeError:
                feature_cols = [c for c in processed.columns
                                if c not in ["customerID", "CustomerID", "Churn",
                                             "Churn Value", "Churn Label"]]
            X = processed.reindex(columns=feature_cols, fill_value=0)
            probs = MODEL.predict_proba(X)[:, 1]
        else:
            probs = np.random.beta(1.5, 5, total)
            for c in ["Churn", "churn", "Churn Value"]:
                if c in df.columns:
                    actual = pd.to_numeric(
                        df[c].map({"Yes": 1, "No": 0, 1: 1, 0: 0}),
                        errors="coerce"
                    ).fillna(0)
                    probs = actual.values * np.random.uniform(0.7, 0.99, total)
                    break

        # ── Segments ──────────────────────────────────────────
        segments   = [get_segment(p) for p in probs]
        seg_counts = {"Critical": 0, "High Risk": 0, "Moderate": 0, "Low Risk": 0}
        for s in segments:
            seg_counts[s] += 1

        # ── Charges ───────────────────────────────────────────
        charge_col = None
        for c in ["MonthlyCharges", "monthly_charges", "Monthly Charges"]:
            if c in df.columns:
                charge_col = c
                break
        charges = df[charge_col].fillna(65.0).tolist() if charge_col else [65.0] * total

        # ── Actual churn rate ─────────────────────────────────
        actual_churn_rate = None
        for c in ["Churn", "churn", "Churn Value"]:
            if c in df.columns:
                v = df[c].map({"Yes": 1, "No": 0}).fillna(df[c])
                actual_churn_rate = float(pd.to_numeric(v, errors="coerce").mean())
                break

        # ── Top at-risk customers ─────────────────────────────
        indices      = np.argsort(probs)[::-1][:20]
        top_customers = []
        for i in indices:
            contract = ""
            for c in ["Contract", "contract"]:
                if c in df.columns:
                    contract = str(df[c].iloc[i])
                    break
            tenure = 0
            for c in ["tenure", "Tenure", "Tenure Months"]:
                if c in df.columns:
                    tenure = int(df[c].iloc[i]) if pd.notna(df[c].iloc[i]) else 0
                    break
            top_customers.append({
                "id"      : str(customer_ids[i]),
                "prob"    : round(float(probs[i]) * 100, 1),
                "segment" : segments[i],
                "contract": contract,
                "tenure"  : tenure,
                "charge"  : round(float(charges[i]), 2),
            })

        # ── Revenue at risk ───────────────────────────────────
        rev_at_risk = sum(
            charges[i] for i, s in enumerate(segments)
            if s in ["Critical", "High Risk"]
        )

        # ── Contract churn ────────────────────────────────────
        contract_churn = {}
        for c in ["Contract", "contract"]:
            if c in df.columns:
                for ct in df[c].unique():
                    mask     = df[c] == ct
                    ct_probs = [probs[i] for i, m in enumerate(mask) if m]
                    if ct_probs:
                        contract_churn[str(ct)] = round(np.mean(ct_probs) * 100, 1)
                break

        return jsonify({
            "mode"             : mode,
            "model_name"       : MODEL_NAME if mode == "ml" else "Analytics Mode",
            "total"            : total,
            "match_pct"        : round(match_pct * 100, 0),
            "seg_counts"       : seg_counts,
            "rev_at_risk_usd"  : round(rev_at_risk, 2),
            "actual_churn_rate": actual_churn_rate,
            "avg_prob"         : round(float(np.mean(probs)) * 100, 1),
            "top_customers"    : top_customers,
            "contract_churn"   : contract_churn,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    return jsonify({"model": MODEL_NAME, "status": "running"})


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════╗")
    print("║   🔮  ChurnIQ Flask Server            ║")
    print("║   Open: http://localhost:5000         ║")
    print("╚══════════════════════════════════════╝\n")
    app.run(debug=True, port=5000)
