"""
src/models/segment.py
─────────────────────
Segments customers by churn risk, generates targeted retention strategies,
and produces an executive-ready insights report mapped to the IBM layout.
"""

import os
import sys
import warnings
from pathlib import Path

# Force Python lookup sync from project workspace root anchor directory
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    AVG_CUSTOMER_LTV,
    DATA_PROCESSED,
    ID_COL,
    MODELS_DIR,
    REPORTS_DIR,
    RETENTION_CAMPAIGN_COST,
    RETENTION_SUCCESS_RATE,
    TARGET_COL,
)

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
WARNING = "#ffd166"
PURPLE = "#a78bfa"

# Risk Segmentation Thresholds
RISK_BINS = [0, 0.3, 0.6, 0.8, 1.01]
RISK_LABELS = ["🟢 Low Risk", "🟡 Moderate", "🟠 High Risk", "🔴 Critical"]

SEGMENT_COLORS = {
    "🟢 Low Risk": "#00d4aa",
    "🟡 Moderate": "#ffd166",
    "🟠 High Risk": "#ff8c42",
    "🔴 Critical": "#ff4b6e",
}

RETENTION_PLAYBOOKS = {
    "🔴 Critical": {
        "label": "Immediate Intervention",
        "actions": [
            "🚨 Personal call from Customer Success within 24 hours",
            "💰 Offer 30% discount or contract upgrade",
            "🎁 Free premium feature unlock for 3 months",
            "📞 Escalate to senior account manager",
        ],
        "priority": "P0",
        "timeline": "24 hours",
        "budget": "$80–150 per customer",
    },
    "🟠 High Risk": {
        "label": "Proactive Retention",
        "actions": [
            "📧 Personalised email campaign highlighting their usage ROI",
            "💡 Offer annual plan with 15% discount",
            "🔧 Assign dedicated onboarding/support specialist",
            "📊 Share personalised value report",
        ],
        "priority": "P1",
        "timeline": "3–5 days",
        "budget": "$30–60 per customer",
    },
    "🟡 Moderate": {
        "label": "Nurture Campaign",
        "actions": [
            "📘 Educational drip campaign (tips, tutorials)",
            "🎯 In-app nudge for underused features",
            "⭐ NPS survey + follow-up on detractors",
            "🤝 Invite to user community / webinar",
        ],
        "priority": "P2",
        "timeline": "1–2 weeks",
        "budget": "$10–20 per customer",
    },
    "🟢 Low Risk": {
        "label": "Loyalty & Upsell",
        "actions": [
            "🏆 Loyalty rewards / referral programme",
            "📈 Upsell to higher-tier plan",
            "🌟 Feature beta access / ambassador programme",
            "✉️ Quarterly value digest newsletter",
        ],
        "priority": "P3",
        "timeline": "Monthly",
        "budget": "$5–10 per customer",
    },
}


def load_and_predict():
    model_data = joblib.load(MODELS_DIR / "best_model.pkl")
    model = model_data["model"]
    model_name = model_data["name"]

    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    cleaned = pd.read_csv(DATA_PROCESSED / "cleaned_data.csv")

    test_idx = y_test.index
    raw_subset = cleaned.iloc[test_idx].reset_index(drop=True)

    probs = model.predict_proba(X_test)[:, 1]

    df = raw_subset.copy()
    df["churn_probability"] = probs
    df["risk_segment"] = pd.cut(
        probs, bins=RISK_BINS, labels=RISK_LABELS, right=False
    )
    df["actual_churn"] = y_test.values

    print(f"🤖 Active Model Context: {model_name}")
    print(f"📊 Scored {len(df):,} customer accounts safely.")
    return df, model_name


def build_segment_summary(df):
    summary = (
        df.groupby("risk_segment", observed=True)
        .agg(
            customers=(ID_COL, "count"),
            avg_churn_prob=("churn_probability", "mean"),
            actual_churn_rate=("actual_churn", "mean"),
            avg_monthly_rev=("Monthly Charges", "mean"),
            avg_tenure=("Tenure Months", "mean"),
            total_at_risk_revenue=("Monthly Charges", "sum"),
        )
        .reset_index()
    )

    summary["est_churn_count"] = (
        summary["customers"] * summary["actual_churn_rate"]
    ).astype(int)
    summary["est_revenue_at_risk"] = (
        summary["est_churn_count"] * summary["avg_monthly_rev"] * 12
    )
    summary["campaign_cost"] = summary["customers"] * RETENTION_CAMPAIGN_COST
    summary["est_saved"] = (
        summary["est_churn_count"]
        * RETENTION_SUCCESS_RATE
        * AVG_CUSTOMER_LTV
    )
    summary["net_roi"] = summary["est_saved"] - summary["campaign_cost"]

    return summary


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_risk_distribution(df, ax):
    counts = df["risk_segment"].value_counts()[RISK_LABELS]
    colors = [SEGMENT_COLORS[l] for l in RISK_LABELS]
    bars = ax.bar(RISK_LABELS, counts.values, color=colors, width=0.6)
    for bar, val in zip(bars, counts.values):
        pct = val / len(df)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}\n{pct:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    ax.set_title("Customer Risk Distribution Matrix", fontsize=13, color="white", pad=12)
    ax.set_ylabel("Number of Accounts")
    ax.tick_params(axis="x", labelsize=8)


def plot_revenue_at_risk(summary, ax):
    colors = [SEGMENT_COLORS[s] for s in summary["risk_segment"]]
    bars = ax.barh(
        summary["risk_segment"],
        summary["est_revenue_at_risk"] / 1000,
        color=colors,
        alpha=0.85,
    )
    for bar, val in zip(bars, summary["est_revenue_at_risk"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}",
            va="center",
            fontsize=9,
            color="white",
        )
    ax.set_title("Estimated Corporate Annual Revenue at Risk", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Revenue exposure threshold ($k)")


def plot_roi_by_segment(summary, ax):
    colors = [ACCENT if r > 0 else DANGER for r in summary["net_roi"]]
    bars = ax.bar(
        summary["risk_segment"],
        summary["net_roi"] / 1000,
        color=colors,
        width=0.6,
    )
    ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, summary["net_roi"]):
        sign = "+" if val > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (1 if val >= 0 else -3),
            f"{sign}${val/1000:.1f}k",
            ha="center",
            va="bottom",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.set_title("Net Strategy ROI Forecast per Segment", fontsize=13, color="white", pad=12)
    ax.set_ylabel("Net Campaign ROI Yield ($k)")
    ax.tick_params(axis="x", labelsize=8)


def plot_churn_prob_dist(df, ax):
    for label in RISK_LABELS:
        subset = df[df["risk_segment"] == label]["churn_probability"]
        if len(subset):
            ax.hist(
                subset,
                bins=30,
                alpha=0.65,
                label=label,
                color=SEGMENT_COLORS[label],
            )
    ax.set_title("Churn Probability Distribution Vectors", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Calibrated Risk Probability")
    ax.set_ylabel("Accounts Registry")
    ax.legend(fontsize=8)


def plot_segment_profile(df, ax):
    # Map feature checks dynamically to updated dataset properties
    profile = df.groupby("risk_segment", observed=True)[
        ["Tenure Months", "Monthly Charges"]
    ].mean()

    x = np.arange(len(profile.columns))
    w = 0.25
    for i, (label, row) in enumerate(profile.iterrows()):
        norm = row / profile.max()
        ax.bar(
            x + i * w,
            norm,
            w,
            label=label,
            color=SEGMENT_COLORS[label],
            alpha=0.85,
        )
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(["Account Lifespan\nTenure", "Monthly Billing\nCharges"], fontsize=9)
    ax.set_title("Normalized Attribute Profile Matrix", fontsize=13, color="white", pad=12)
    ax.set_ylabel("Relative Scale Weights")
    ax.legend(fontsize=7)


def plot_contract_segment(df, ax):
    ct = (
        df.groupby(["Contract", "risk_segment"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    colors = [SEGMENT_COLORS[l] for l in ct_pct.columns]
    ct_pct.plot(kind="bar", ax=ax, color=colors, stacked=True, width=0.6, rot=0)
    ax.set_title("Risk Concentration Split by Contract Type", fontsize=13, color="white", pad=12)
    ax.set_ylabel("Proportion % of Accounts Group")
    ax.legend(fontsize=7, loc="upper right")


def generate_segment_report(df, summary):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 20), facecolor="#0f1117")
    fig.suptitle(
        "Customer Risk Segmentation & Retention Strategy Metrics",
        fontsize=20,
        color="white",
        fontweight="bold",
        y=0.99,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    plot_risk_distribution(df, axes[0])
    plot_revenue_at_risk(summary, axes[1])
    plot_roi_by_segment(summary, axes[2])
    plot_churn_prob_dist(df, axes[3])
    plot_segment_profile(df, axes[4])
    plot_contract_segment(df, axes[5])

    out = REPORTS_DIR / "segmentation_report.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"📊 Graphical Strategy Performance Matrix Matrix Saved → {out}")


# ── Text Retention Playbook ───────────────────────────────────────────────────
def generate_retention_playbook(df, summary):
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║         CUSTOMER RETENTION MANAGEMENT STRATEGY PLAYBOOK       ║",
        "╚══════════════════════════════════════════════════════════════╝\n",
    ]
    total_revenue_at_risk = summary["est_revenue_at_risk"].sum()
    total_roi = summary["net_roi"].sum()

    lines += [
        f"💼 EXECUTIVE PORTFOLIO SUMMARY",
        f"   Total customer accounts evaluated : {len(df):,}",
        f"   Total yearly core revenue at risk : ${total_revenue_at_risk:,.0f}/year",
        f"   Projected net campaign ROI yield  : ${total_roi:,.0f}",
        f"   Campaign portfolio break-even rate: {RETENTION_CAMPAIGN_COST / AVG_CUSTOMER_LTV:.1%}\n",
    ]

    for _, row in summary.iterrows():
        seg = row["risk_segment"]
        playbook = RETENTION_PLAYBOOKS.get(seg, {})
        lines += [
            f"\n{'─'*60}",
            f"  {seg} [{playbook.get('priority','')}] — {playbook.get('label','')}",
            f"{'─'*60}",
            f"  Target Volume    : {row['customers']:,} accounts",
            f"  Mean Churn Prob  : {row['avg_churn_prob']:.1%}",
            f"  Expected Churners: {row['est_churn_count']:,} cases",
            f"  Revenue Exposure : ${row['est_revenue_at_risk']:,.0f}/year",
            f"  Campaign Capital : ${row['campaign_cost']:,.0f}",
            f"  Est. Saved Asset : ${row['est_saved']:,.0f}",
            f"  Net Metric Yield : ${row['net_roi']:,.0f}",
            f"  Timeline Horizon : {playbook.get('timeline','')}",
            f"  Budget Allocation: {playbook.get('budget','')}",
            f"\n  📋 Strategic Mitigation Actions:",
        ]
        for action in playbook.get("actions", []):
            lines.append(f"     {action}")

    text = "\n".join(lines)
    print(text)
    (REPORTS_DIR / "retention_playbook.txt").write_text(text, encoding="utf-8")
    return text


# ── Export at-risk customer list ──────────────────────────────────────────────
def export_at_risk_list(df):
    at_risk = df[df["risk_segment"].isin(["🔴 Critical", "🟠 High Risk"])].copy()
    at_risk = at_risk[
        [
            ID_COL,
            "churn_probability",
            "risk_segment",
            "Contract",
            "Tenure Months",
            "Monthly Charges",
        ]
    ].sort_values("churn_probability", ascending=False)
    out = REPORTS_DIR / "at_risk_customers.csv"
    at_risk.to_csv(out, index=False)
    print(f"\n📋 Priority CRM outreach manifest saved safely to → {out} ({len(at_risk):,} accounts)")
    return at_risk


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    df, model_name = load_and_predict()
    summary = build_segment_summary(df)
    generate_segment_report(df, summary)
    generate_retention_playbook(df, summary)
    export_at_risk_list(df)
    print("\n✅ Stratification & operations routing engine complete!")
    return df, summary


if __name__ == "__main__":
    run()