"""
CO2 Storage Site Suitability — ML Training Pipeline
=====================================================
Trains two models:
  1. XGBoostClassifier  → binary suitability label (suitable / unsuitable)
  2. XGBoostRegressor   → continuous risk score (0–1)

Outputs serialized models + a JSON summary of evaluation metrics.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, mean_squared_error, mean_absolute_error, r2_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from xgboost                   import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "porosity", "permeability_md", "depth_m", "pressure_mpa",
    "temperature_c", "caprock_thickness_m", "caprock_integrity",
    "salinity_g_l", "trap_area_km2", "fault_distance_km",
    "geothermal_gradient", "pressure_depth_ratio",
]

RANDOM_SEED = 42


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_splits():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val   = pd.read_csv(f"{DATA_DIR}/val.csv")
    test  = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, val, test


def get_Xy(df, target):
    return df[FEATURES].values, df[target].values


# ── 2. Build pipelines ───────────────────────────────────────────────────────
def build_classifier():
    """XGBoost classifier with scale_pos_weight to handle class imbalance."""
    clf = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        use_label_encoder  = False,
        eval_metric        = "logloss",
        random_state       = RANDOM_SEED,
        n_jobs             = -1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_regressor():
    """XGBoost regressor for continuous risk score prediction."""
    reg = XGBRegressor(
        n_estimators       = 400,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 3,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        eval_metric        = "rmse",
        random_state       = RANDOM_SEED,
        n_jobs             = -1,
    )
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


# ── 3. Train ─────────────────────────────────────────────────────────────────
def train_classifier(pipe, X_train, y_train, X_val, y_val):
    pipe.fit(
        X_train, y_train,
        clf__eval_set=[(X_val, y_val)],
        clf__verbose=False,
    )
    return pipe


def train_regressor(pipe, X_train, y_train, X_val, y_val):
    pipe.fit(
        X_train, y_train,
        reg__eval_set=[(X_val, y_val)],
        reg__verbose=False,
    )
    return pipe


# ── 4. Evaluate ──────────────────────────────────────────────────────────────
def evaluate_classifier(pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print("\n── Classifier Evaluation ────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Unsuitable", "Suitable"]))
    print(f"  ROC-AUC : {auc:.4f}")

    return {"report": report, "auc": auc, "cm": cm, "fpr": fpr, "tpr": tpr, "y_proba": y_proba}


def evaluate_regressor(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print("\n── Regressor Evaluation ─────────────────────────────────")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


# ── 5. Cross-validation ──────────────────────────────────────────────────────
def cross_validate_classifier(pipe, X, y):
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n  5-Fold CV ROC-AUC : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── 6. Feature importance ────────────────────────────────────────────────────
def get_feature_importance(pipe, model_key="clf"):
    model = pipe.named_steps[model_key]
    imp   = model.feature_importances_
    return pd.Series(imp, index=FEATURES).sort_values(ascending=False)


# ── 7. Plotting ───────────────────────────────────────────────────────────────
def plot_all(clf_pipe, reg_pipe, clf_eval, reg_eval, X_test, y_cls_test, y_risk_test):
    """Produce a single comprehensive evaluation figure."""
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT   = "#00e5ff"
    ACCENT2  = "#ff6b6b"
    GRID_CLR = "#2a2a3e"
    TXT_CLR  = "#e0e0e0"

    def style_ax(ax, title=""):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=TXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TXT_CLR)
        ax.yaxis.label.set_color(TXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.5)
        if title:
            ax.set_title(title, color=TXT_CLR, fontsize=11, fontweight="bold", pad=8)

    # ── (0,0) ROC Curve ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "ROC Curve — Classifier")
    ax0.plot(clf_eval["fpr"], clf_eval["tpr"], color=ACCENT, lw=2,
             label=f'AUC = {clf_eval["auc"]:.3f}')
    ax0.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax0.set_xlabel("False Positive Rate"); ax0.set_ylabel("True Positive Rate")
    ax0.legend(facecolor="#1a1a2e", edgecolor=GRID_CLR, labelcolor=TXT_CLR, fontsize=9)

    # ── (0,1) Confusion Matrix ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "Confusion Matrix")
    cm_norm = clf_eval["cm"].astype(float) / clf_eval["cm"].sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax1, cmap="Blues",
                xticklabels=["Unsuitable", "Suitable"],
                yticklabels=["Unsuitable", "Suitable"],
                cbar=False, linewidths=0.5)
    ax1.set_xlabel("Predicted", color=TXT_CLR); ax1.set_ylabel("Actual", color=TXT_CLR)
    ax1.tick_params(colors=TXT_CLR)
    for text in ax1.texts:
        text.set_color(TXT_CLR)

    # ── (0,2) Prediction confidence distribution ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "Predicted Probability Distribution")
    for label, color, name in [(0, ACCENT2, "Unsuitable"), (1, ACCENT, "Suitable")]:
        mask = y_cls_test == label
        ax2.hist(clf_eval["y_proba"][mask], bins=30, alpha=0.6,
                 color=color, label=name, density=True)
    ax2.set_xlabel("P(Suitable)"); ax2.set_ylabel("Density")
    ax2.legend(facecolor="#1a1a2e", edgecolor=GRID_CLR, labelcolor=TXT_CLR, fontsize=9)

    # ── (1,0–1) Feature importance — Classifier ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, "Feature Importance — Suitability Classifier")
    fi_clf = get_feature_importance(clf_pipe, "clf")
    colors = [ACCENT if i < 3 else "#4a9eff" for i in range(len(fi_clf))]
    bars = ax3.barh(fi_clf.index[::-1], fi_clf.values[::-1], color=colors[::-1], edgecolor="none")
    ax3.set_xlabel("Importance Score")

    # ── (1,2) Feature importance — Regressor ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "Feature Importance — Risk Regressor")
    fi_reg = get_feature_importance(reg_pipe, "reg").head(7)
    ax4.barh(fi_reg.index[::-1], fi_reg.values[::-1], color=ACCENT2, edgecolor="none")
    ax4.set_xlabel("Importance Score")

    # ── (2,0) Risk: actual vs predicted ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, "Risk Score: Actual vs Predicted")
    ax5.scatter(y_risk_test, reg_eval["y_pred"], alpha=0.3, s=12, color=ACCENT2)
    mn, mx = 0, 1
    ax5.plot([mn, mx], [mn, mx], "--", color="#888", lw=1)
    ax5.set_xlabel("Actual Risk Score"); ax5.set_ylabel("Predicted Risk Score")
    ax5.text(0.05, 0.92, f'R² = {reg_eval["r2"]:.3f}\nRMSE = {reg_eval["rmse"]:.3f}',
             transform=ax5.transAxes, color=TXT_CLR, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f1117", edgecolor=GRID_CLR))

    # ── (2,1) Risk distribution by suitability class ─────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, "Risk Score Distribution by Class")
    for label, color, name in [(0, ACCENT2, "Unsuitable"), (1, ACCENT, "Suitable")]:
        mask = y_cls_test == label
        ax6.hist(reg_eval["y_pred"][mask], bins=30, alpha=0.6,
                 color=color, label=name, density=True)
    ax6.set_xlabel("Predicted Risk Score"); ax6.set_ylabel("Density")
    ax6.legend(facecolor="#1a1a2e", edgecolor=GRID_CLR, labelcolor=TXT_CLR, fontsize=9)

    # ── (2,2) Metrics summary card ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor("#1a1a2e")
    for spine in ax7.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax7.set_xticks([]); ax7.set_yticks([])
    ax7.set_title("Model Metrics Summary", color=TXT_CLR, fontsize=11, fontweight="bold", pad=8)

    rpt   = clf_eval["report"]
    lines = [
        ("── Classifier ──────", "#888"),
        (f"  ROC-AUC     {clf_eval['auc']:.4f}", ACCENT),
        (f"  Precision   {rpt['1']['precision']:.4f}", TXT_CLR),
        (f"  Recall      {rpt['1']['recall']:.4f}", TXT_CLR),
        (f"  F1-Score    {rpt['1']['f1-score']:.4f}", TXT_CLR),
        (f"  Accuracy    {rpt['accuracy']:.4f}", TXT_CLR),
        ("", ""),
        ("── Risk Regressor ──", "#888"),
        (f"  R²          {reg_eval['r2']:.4f}", ACCENT2),
        (f"  RMSE        {reg_eval['rmse']:.4f}", TXT_CLR),
        (f"  MAE         {reg_eval['mae']:.4f}", TXT_CLR),
    ]
    for i, (txt, clr) in enumerate(lines):
        if not clr:
            continue
        ax7.text(0.08, 0.93 - i * 0.085, txt, transform=ax7.transAxes,
                 color=clr, fontsize=9.5, family="monospace")

    fig.suptitle("CO₂ Storage Site Suitability — Model Evaluation Dashboard",
                 color="white", fontsize=15, fontweight="bold", y=0.99)

    out_path = f"{OUTPUT_DIR}/evaluation_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Dashboard saved → {out_path}")
    return out_path


# ── 8. Save artifacts ────────────────────────────────────────────────────────
def save_artifacts(clf_pipe, reg_pipe, metrics):
    joblib.dump(clf_pipe, f"{OUTPUT_DIR}/classifier_pipeline.joblib")
    joblib.dump(reg_pipe, f"{OUTPUT_DIR}/regressor_pipeline.joblib")
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Models saved → {OUTPUT_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CO₂ Storage Site Suitability — Training Pipeline")
    print("=" * 60)

    # 1. Load
    train, val, test = load_splits()
    X_train, y_cls_train  = get_Xy(train, "suitable")
    X_val,   y_cls_val    = get_Xy(val,   "suitable")
    X_test,  y_cls_test   = get_Xy(test,  "suitable")

    y_risk_train = train["risk_score"].values
    y_risk_val   = val["risk_score"].values
    y_risk_test  = test["risk_score"].values

    # 2. Classifier
    print("\n[1/4] Training XGBoost Classifier …")
    clf_pipe = build_classifier()
    clf_pipe = train_classifier(clf_pipe, X_train, y_cls_train, X_val, y_cls_val)

    print("[2/4] Cross-validating classifier …")
    cv_scores = cross_validate_classifier(clf_pipe, X_train, y_cls_train)

    # 3. Regressor
    print("\n[3/4] Training XGBoost Risk Regressor …")
    reg_pipe = build_regressor()
    reg_pipe = train_regressor(reg_pipe, X_train, y_risk_train, X_val, y_risk_val)

    # 4. Evaluate
    print("\n[4/4] Evaluating on held-out test set …")
    clf_eval = evaluate_classifier(clf_pipe, X_test, y_cls_test)
    reg_eval = evaluate_regressor(reg_pipe, X_test, y_risk_test)

    # 5. Plot
    plot_all(clf_pipe, reg_pipe, clf_eval, reg_eval, X_test, y_cls_test, y_risk_test)

    # 6. Save
    metrics = {
        "classifier": {
            "roc_auc":   round(clf_eval["auc"], 4),
            "accuracy":  round(clf_eval["report"]["accuracy"], 4),
            "f1_suitable": round(clf_eval["report"]["1"]["f1-score"], 4),
            "precision": round(clf_eval["report"]["1"]["precision"], 4),
            "recall":    round(clf_eval["report"]["1"]["recall"], 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std":  round(cv_scores.std(), 4),
        },
        "regressor": {
            "rmse": round(reg_eval["rmse"], 4),
            "mae":  round(reg_eval["mae"], 4),
            "r2":   round(reg_eval["r2"], 4),
        }
    }
    save_artifacts(clf_pipe, reg_pipe, metrics)

    print("\n" + "=" * 60)
    print("  Training complete ✓")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    main()
