"""
src/database/run_analytics.py — SQL Relational Corporate Intelligence Analytics Engine
"""
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Lock central system configurations anchors securely
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PROCESSED, TARGET_COL

def execute_sql_analytics():
    print("\n🗄️ Running Advanced SQL Relational Revenue Analytics Queries...")
    
    db_path = "telecom_corporate.db"
    conn = sqlite3.connect(db_path)
    
    # Load cleaned data cleanly to build relational local tables
    cleaned_data_path = DATA_PROCESSED / "cleaned_data.csv"
    if not cleaned_data_path.exists():
        print(f"❌ Error: Cleaned engine file records missing at {cleaned_data_path}")
        return
        
    df = pd.read_csv(cleaned_data_path)
    
    # Write into local SQLite instance replacing previous structural iterations
    df.to_sql("customers", conn, if_exists="replace", index=False)
    
    # ── QUERY 1: Internet Service vs Revenue Leaks (Mapped to IBM Layout) ──
    print("\n--- 📊 SQL RESULT: PRODUCT & INTERNET SERVICE LINE SEGMENTATION ---")
    query_1 = f"""
        SELECT 
            `Internet Service` as InternetServiceLine,
            COUNT(*) as TotalCustomers,
            SUM(`{TARGET_COL}`) as ChurnCount,
            ROUND(AVG(`{TARGET_COL}`) * 100, 2) as ChurnRatePct,
            ROUND(SUM(CASE WHEN `{TARGET_COL}` = 1 THEN `Monthly Charges` ELSE 0 END), 2) as LostMonthlyRevenue
        FROM customers
        GROUP BY `Internet Service`
        ORDER BY LostMonthlyRevenue DESC;
    """
    try:
        print(pd.read_sql_query(query_1, conn).to_string(index=False))
    except Exception as e:
        print(f"⚠️ Query 1 execution warning: {e}")

    # ── QUERY 2: Financial Revenue Mapped across High-Risk Contracts ──
    print("\n--- 💼 SQL RESULT: STRATEGIC CONTRACT FINANCIAL LOSS CHANNEL MATRIX ---")
    query_2 = f"""
        SELECT 
            Contract,
            COUNT(*) as TotalAccounts,
            SUM(`{TARGET_COL}`) as HighRiskChurners,
            ROUND(AVG(`Monthly Charges`), 2) as MeanMonthlyInvoice,
            ROUND(SUM(CASE WHEN `{TARGET_COL}` = 1 THEN `Total Charges` ELSE 0 END), 2) as CumulativeVulnerableCapital
        FROM customers
        GROUP BY Contract
        ORDER BY HighRiskChurners DESC;
    """
    try:
        print(pd.read_sql_query(query_2, conn).to_string(index=False))
    except Exception as e:
        print(f"⚠️ Query 2 execution warning: {e}")

    conn.close()
    print("\n✅ SQL Relational Corporate Analytics processing completed successfully!")

if __name__ == "__main__":
    execute_sql_analytics()