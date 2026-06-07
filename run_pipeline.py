"""
run_pipeline.py
───────────────
Master runner — executes the full end-to-end pipeline in one command.
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🔮  CHURN PREDICTION & RETENTION SYSTEM                    ║
║       End-to-End Full-Stack ML Pipeline                      ║
╚══════════════════════════════════════════════════════════════╝
"""

def run(data_file: str = "customer_data.csv"):
    print(BANNER)
    t0 = time.time()

    # ── Step 0: Generate sample data if needed ────────────────
    raw_path = Path("data/raw") / data_file
    if not raw_path.exists():
        print("📦 STEP 0 — Generating sample data (no file found)...")
        from data.generate_sample_data import generate
        generate()
    else:
        print(f"📂 STEP 0 — Using existing data: {raw_path}")

    # ── Step 1: Preprocessing ─────────────────────────────────
    print("\n" + "─"*55)
    print("🔧 STEP 1 — Data Preprocessing & Feature Engineering")
    print("─"*55)
    from src.data.preprocess import run_pipeline
    run_pipeline(data_file)

    # ── Step 2: EDA ───────────────────────────────────────────
    print("\n" + "─"*55)
    print("📊 STEP 2 — Exploratory Data Analysis (Automated Plots)")
    print("─"*55)
    from src.features.eda import generate_eda_report, generate_text_summary
    from config import DATA_PROCESSED
    import pandas as pd
    df = pd.read_csv(DATA_PROCESSED / "cleaned_data.csv")
    generate_text_summary(df)
    generate_eda_report(df)

    # ── Step 3: Model Training ────────────────────────────────
    print("\n" + "─"*55)
    print("🤖 STEP 3 — Model Training & Matrix Calibration")
    print("─"*55)
    from src.models.train import run as train_run
    train_run()

    # ── Step 3.5: SQL Database Layer ──────────────────────────
    print("\n" + "─"*55)
    print("🗄️  STEP 3.5 — SQL Relational Database Analysis")
    print("─"*55)
    try:
        from src.database.run_analytics import execute_sql_analytics
        execute_sql_analytics()
    except ImportError:
        print("⚠️  SQL database script not executed. Ensure src/database/run_analytics.py exists.")

    # ── Step 4: Segmentation & Strategy ──────────────────────
    print("\n" + "─"*55)
    print("🎯 STEP 4 — Customer Segmentation & Retention Strategy")
    print("─"*55)
    from src.models.segment import run as segment_run
    segment_run()

    # ── Done ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅  PIPELINE COMPLETE  ({elapsed:.1f}s)                              ║
╠══════════════════════════════════════════════════════════════╣
║  📁 OUTPUTS GENERATED                                         ║
║     data/processed/      → cleaned data + train/test splits  ║
║     src/models/          → best_model.pkl                     ║
║     reports/             → eda_report.png & model_evaluation.png ║
║     reports/retention_playbook.txt  → operational strategies ║
║     reports/at_risk_customers.csv   → automated CRM export    ║
║     dashboard/executive_dashboard.html  → live UI tracker    ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")
    parser.add_argument("--data", default="customer_data.csv",
                        help="CSV filename inside data/raw/ (default: customer_data.csv)")
    args = parser.parse_args()
    run(args.data)