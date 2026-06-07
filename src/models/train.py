"""
src/models/train.py
───────────────────
Trains, evaluates, and saves multiple ML models for the IBM Churn Dataset.
Produces: model comparisons, ROC curve, confusion matrix, feature importance.
"""

import os
import sys
import warnings
from pathlib import Path

# Fix path indexing dynamically across windows workspace folders
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
warnings.filterwarnings("ignore")

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from config import (
    AVG_CUSTOMER_LTV,
    CLASSIFICATION_THRESHOLD,
    CV_FOLDS,
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RETENTION_CAMPAIGN_COST,
    RETENTION_SUCCESS_RATE,
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


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    print(f"📂 Loaded Train Split: {X_train.shape} | Test Split: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Define models ─────────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


# ── Train & evaluate all models ───────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    for name, model in models.items():
        print(f"\n🤖 Training evaluation matrix for: {name}...")
        model.fit(X_train, y_train)

        # Calculate exact probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

        # Cross Validation Evaluation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        metrics = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "avg_prec": average_precision_score(y_test, y_prob),
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
        results[name] = metrics
        print(
            f"   AUC: {metrics['roc_auc']:.4f}  |  F1: {metrics['f1']:.4f}  "
            f"|  CV-AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}"
        )

    return results


# ── Select best model ─────────────────────────────────────────────────────────
def select_best(results):
    best = max(results.items(), key=lambda x: x[1]["roc_auc"])
    print(f"\n🏆 Best model selection verified: {best[0]} (AUC={best[1]['roc_auc']:.4f})")
    return best[0], best[1]


# ── Save best model ───────────────────────────────────────────────────────────
def save_model(name, model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"name": name, "model": model}, path)
    print(f"💾 Model saved to binary checkpoint → {path}")


# ── Plot: Model Comparison ────────────────────────────────────────────────────
def plot_model_comparison(results, ax):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.18
    colors = [ACCENT, DANGER, WARNING, PURPLE]
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        ax.bar(
            x + i * width, vals, width, label=name, color=colors[i], alpha=0.85
        )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison", fontsize=13, color="white", pad=12)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.axhline(0.8, color="white", linestyle="--", alpha=0.3)


# ── Plot: ROC Curves ──────────────────────────────────────────────────────────
def plot_roc_curves(results, y_test, ax):
    colors = [ACCENT, DANGER, WARNING, PURPLE]
    for (name, res), col in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(
            fpr,
            tpr,
            color=col,
            linewidth=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, label="Random")
    ax.set_title("ROC Curves — All Models", fontsize=13, color="white", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(fontsize=8)


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(best_name, best_res, y_test, ax):
    cm = confusion_matrix(y_test, best_res["y_pred"])
    pct = cm / cm.sum() * 100
    labels = np.array(
        [
            [f"{v:,}\n({p:.1f}%)" for v, p in zip(row, prow)]
            for row, prow in zip(cm, pct)
        ]
    )
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        ax=ax,
        cmap="RdYlGn",
        linewidths=0.5,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        annot_kws={"size": 11},
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Predicted Label Target")
    ax.set_ylabel("Actual Label Target")


# ── Plot: Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(best_name, best_model, X_train, ax, top_n=15):
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X_train.columns)
    else:
        # Secure single-dimension vector extraction to bypass coefficient dimension warnings
        fi = pd.Series(np.abs(best_model.coef_[0].flatten()), index=X_train.columns)

    fi = fi.nlargest(top_n).sort_values()
    colors = [DANGER if v > fi.median() else ACCENT for v in fi.values]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top {top_n} Feature Importances — {best_name}", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Importance Weight Metric")


# ── Plot: Precision-Recall ────────────────────────────────────────────────────
def plot_pr_curve(best_name, best_res, y_test, ax):
    prec, rec, _ = precision_recall_curve(y_test, best_res["y_prob"])
    ap = best_res["avg_prec"]
    ax.plot(rec, prec, color=ACCENT, linewidth=2, label=f"AP={ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.1, color=ACCENT)
    ax.axhline(
        y_test.mean(),
        color=DANGER,
        linestyle="--",
        alpha=0.7,
        label=f"Baseline ({y_test.mean():.2f})",
    )
    ax.set_title(f"Precision-Recall Curve — {best_name}", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()


# ── Plot: Business ROI ────────────────────────────────────────────────────────
def plot_business_roi(best_res, y_test, ax):
    thresholds = np.linspace(0.1, 0.9, 50)
    rois = []
    for t in thresholds:
        preds = (best_res["y_prob"] >= t).astype(int)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        saved = tp * RETENTION_SUCCESS_RATE * AVG_CUSTOMER_LTV
        cost = (tp + fp) * RETENTION_CAMPAIGN_COST
        roi = saved - cost
        rois.append(roi)

    best_t = thresholds[np.argmax(rois)]
    best_roi = max(rois)

    ax.plot(thresholds * 100, rois, color=WARNING, linewidth=2)
    ax.fill_between(
        thresholds * 100, rois, 0, where=np.array(rois) > 0, alpha=0.2, color=ACCENT, label="Profit Zone"
    )
    ax.fill_between(
        thresholds * 100, rois, 0, where=np.array(rois) < 0, alpha=0.2, color=DANGER, label="Loss Zone"
    )
    ax.axvline(
        best_t * 100, color=ACCENT, linestyle="--", linewidth=1.5, label=f"Optimal threshold {best_t:.2f}"
    )
    ax.set_title("Business ROI by Classification Threshold", fontsize=13, color="white", pad=12)
    ax.set_xlabel("Threshold (%)")
    ax.set_ylabel("Net Saving Yield Metric ($)")
    ax.legend(fontsize=8)
    ax.text(
        best_t * 100 + 2, best_roi * 0.9, f"${best_roi:,.0f}", color=ACCENT, fontsize=10, fontweight="bold"
    )


# ── Master evaluation plot ────────────────────────────────────────────────────
def generate_model_report(results, y_test, X_train):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_name, best_res = select_best(results)
    best_model = best_res["model"]

    fig = plt.figure(figsize=(22, 24), facecolor="#0f1117")
    fig.suptitle(
        "Customer Churn — Model Evaluation Performance Report",
        fontsize=20,
        color="white",
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]

    plot_model_comparison(results, axes[0])
    plot_roc_curves(results, y_test, axes[1])
    plot_confusion_matrix(best_name, best_res, y_test, axes[2])
    plot_feature_importance(best_name, best_model, X_train, axes[3])
    plot_pr_curve(best_name, best_res, y_test, axes[4])
    plot_business_roi(best_res, y_test, axes[5])

    out = REPORTS_DIR / "model_evaluation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"📊 Dark-Themed Matrix Report Saved → {out}")
    return best_name, best_model


# ── Classification Report Text ────────────────────────────────────────────────
def print_classification_reports(results, y_test):
    for name, res in results.items():
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        print(
            classification_report(
                y_test, res["y_pred"], target_names=["Retained", "Churned"]
            )
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    X_train, X_test, y_train, y_test = load_data()
    results = train_all(X_train, X_test, y_train, y_test)
    best_name, best_model = generate_model_report(results, y_test, X_train)
    save_model(best_name, best_model)
    print_classification_reports(results, y_test)
    print("\n✅ Model execution lifecycle comparison matrix complete!")
    return results


if __name__ == "__main__":
    run()