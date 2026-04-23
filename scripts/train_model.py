"""
LightGBM Fraud Detection Model - Training Script


Input: 
data/train_rebalanced.csv.gz
data/test_holdout.csv.gz

Output: 
models/lgbm_fraud.pkl           (trained model)
models/feature_cols.json        (ordered feature list)
models/threshold.json           (optimal decision threshold)
models/eval_report.txt          (full evaluation summary)
models/feature_importance.csv   (gain + SHAP importances)
data/test_scored.csv.gz         (holdout with FraudScore + Decision)


Training strategy:
- LightGBM
- scale_pos_weight (n_neg/n_pos) for class imbalance correction
- Optimize for PR-AUC (average_precision) - (superior to  ROC-AUC for severe class imbalance)
- Early stopping on validation PR-AUC (20% of train as internal val)
- Optuna hyperparameter search
- SHAP values computed on a 5k-row sample for interpretability
- Threshold optimized for max F1 on holdout

  
Tasks:
# Train with defaults
python train_model.py
# Train + hyperparameter tuning (slower, better model)
python train_model.py --tune
# Use a specific decision threshold instead of auto-optimizing
python train_model.py --threshold 0.5
# Optimize threshold for recall >= 0.80 instead of max F1
python train_model.py --min-recall 0.80
"""


import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from config import BASE_DIR

# paths
TRAIN_GZ = BASE_DIR / "data" / "train_rebalanced.csv.gz"
TEST_GZ = BASE_DIR / "data" / "test_holdout.csv.gz"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "lgbm_fraud.pkl"
FEAT_PATH = MODELS_DIR / "feature_cols.json"
THRESH_PATH = MODELS_DIR / "threshold.json"
REPORT_PATH = MODELS_DIR / "eval_report.txt"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
SCORED_PATH = BASE_DIR / "data" / "test_scored.csv.gz"
PLOTS_DIR = MODELS_DIR / "plots"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Feature columns

# Columns to exclude from features
EXCLUDE_COLS = {
    "trans_num", 
    "cc_num", 
    "merchant", 
    "is_fraud",
    "unix_time", 
    "cust_lat", 
    "cust_long", 
    "merch_lat", 
    "merch_long",
    # String/categorical cols that need special handling
    "IP_Country", 
    "MerchantCategory", 
    "MerchantCountry",
    "CarrierType", 
    "IdentityVerificationStatus",
}

# Categorical columns
CAT_COLS = [
    "IP_Country", 
    "MerchantCategory", 
    "MerchantCountry",
    "CarrierType", 
    "IdentityVerificationStatus",
    "IsWeekend", 
    "IsNightTxn", 
    "DayOfWeek", 
    "TransactionHour",
]


def get_feature_cols(df):
    """Return ordered list of numeric feature columns."""
    exclude = EXCLUDE_COLS - set(CAT_COLS)   # keep cats for encoding
    cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]
    return cols


def encode_categoricals(df, cat_cols, encoders=None):
    """
    Label-encode categorical columns.
    If encoders provided, apply existing (inference). Else, fit new encoders (training).

    Returns (df_encoded, encoders_dict).
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            if col not in df.columns:
                continue
            cats = pd.Categorical(df[col].fillna("__NA__"))
            df[col] = cats.codes
            encoders[col] = list(cats.categories)
    else:
        for col, cats in encoders.items():
            if col not in df.columns:
                df[col] = -1
                continue
            cat_map = {v: i for i, v in enumerate(cats)}
            df[col] = df[col].fillna("__NA__").map(cat_map).fillna(-1).astype(int)
    return df, encoders


# LightGBM default params

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"], # 
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,  # drop from 127
    "max_depth": 7,  # prevent deep overfitting
    "min_child_samples": 100,  # increase from 50
    "feature_fraction": 0.7,  # reduced from 0.8
    "bagging_fraction": 0.7,  # reduced from 0.8
    "bagging_freq": 5,
    "reg_alpha": 0.5,  # increased L1
    "reg_lambda": 2.0,  # increased L2
    "min_split_gain": 0.01,  # require minimum gain to split
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# Hyperparameter tuning (Optuna)

def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log("[WARN] optuna not installed - skipping tuning. Run: pip install optuna")
        return DEFAULT_PARAMS

    log(f"Hyperparameter tuning ({n_trials} Optuna trials) ...")

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        preds = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        "objective": "binary", 
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt", 
        "verbose": -1, 
        "n_jobs": -1,
        "random_state": 42, 
        "n_estimators": 2000,
    })
    log(f"Best PR-AUC: {study.best_value:.4f}")
    log(f"Best params: {study.best_params}")
    return best


# Threshold optimization

def optimize_threshold(y_true, y_prob, min_recall=None):
    """
    Find threshold that maximizes F1.
    If min_recall is set, constrain to thresholds achieving >= min_recall first.
    """
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    if min_recall is not None:
        mask = rec >= min_recall
        if mask.any():
            idx = np.argmax(f1[mask])
            # Map back to original index
            orig_idx = np.where(mask)[0][idx]
            return float(thresh[orig_idx]), float(f1[orig_idx]), float(prec[orig_idx]), float(rec[orig_idx])

    idx = np.argmax(f1[:-1])   # last element has no threshold
    return float(thresh[idx]), float(f1[idx]), float(prec[idx]), float(rec[idx])


# Evaluation (generate report)

def evaluate(model, X, y, threshold, label="Holdout", feature_cols=None):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    lines = [
        f"\n{'-'*60}",
        f"EVALUATION - {label}",
        f"{'-'*60}",
        f"Threshold       : {threshold:.4f}",
        f"ROC-AUC         : {roc:.4f}",
        f"PR-AUC          : {pr_auc:.4f}",
        f"Precision       : {prec:.4f}",
        f"Recall          : {rec:.4f}",
        f"F1              : {f1:.4f}",
        f"",
        f"Confusion Matrix:",
        f"                Pred Legit  Pred Fraud",
        f"Actual Legit  : {cm[0,0]:>10,}  {cm[0,1]:>10,}",
        f"Actual Fraud  : {cm[1,0]:>10,}  {cm[1,1]:>10,}",
        f"",
        f"Fraud caught (TP)       : {cm[1,1]:,}  / {y.sum():,}  ({cm[1,1]/y.sum():.1%})",
        f"Legit blocked (FP)      : {cm[0,1]:,}  / {(y==0).sum():,}  ({cm[0,1]/(y==0).sum():.3%})",
        f"{'-'*60}",
    ]
    return "\n".join(lines), y_prob


# Feature importance

def compute_shap(model, X_sample, feature_cols, plots_dir):
    log("Computing SHAP values (sample of 5k rows) ...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # LightGBM binary returns list [neg_class, pos_class] or single array
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_cols,"mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)

        # SHAP summary bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feature_cols, plot_type="bar", show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(plots_dir / "shap_importance.png", dpi=120, bbox_inches="tight")
        plt.close()

        # SHAP beeswarm
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feature_cols, show=False, max_display=25)
        plt.tight_layout()

        plt.savefig(plots_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
        plt.close()

        log(f"SHAP plots saved to {plots_dir}")
        return shap_df

    except Exception as e:
        log(f"[WARN] SHAP failed: {e}")
        return None


# PR plot

def plot_pr_curve(y_true, y_prob, threshold, pr_auc, plots_dir):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, color="#e74c3c", linewidth=2, label=f"LightGBM (PR-AUC = {pr_auc:.3f})")
    plt.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3%})")

    # Mark threshold
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmin(np.abs(thresh - threshold)) if len(thresh) > 0 else 0
    plt.scatter(rec[idx], prec[idx], s=120, color="#2c3e50", zorder=5, label=f"Threshold = {threshold:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Holdout Set", fontweight="bold")
    plt.legend()
    plt.tight_layout()

    plt.savefig(plots_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    log(f"PR curve saved -> {plots_dir / 'pr_curve.png'}")



# main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search (50 trials)")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--threshold", type=float, default=None, help="Fix decision threshold (default: auto-optimize for max F1)")
    p.add_argument("--min-recall", type=float, default=None, help="Optimize threshold subject to recall >= this value")
    p.add_argument("--val-frac", type=float, default=0.20, help="Fraction of train used for internal validation (default 0.20)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("-" * 65)
    print("LIGHTGBM FRAUD DETECTION - TRAINING")
    print("-" * 65)

    # Load data
    log(f"Loading training set ...")
    if not TRAIN_GZ.exists():
        print(f"[ERROR] {TRAIN_GZ} not found. Run build_training_set.py first.")
        sys.exit(1)

    train = pd.read_csv(TRAIN_GZ, low_memory=False)
    test = pd.read_csv(TEST_GZ, low_memory=False)
    log(f"  Train: {len(train):,} rows  ({train['is_fraud'].mean():.3%} fraud)")
    log(f"  Test : {len(test):,} rows   ({test['is_fraud'].mean():.3%} fraud)")

    # Encode categoricals
    log("Encoding categorical features ...")
    active_cats = [c for c in CAT_COLS if c in train.columns]
    train, encoders = encode_categoricals(train, active_cats)
    test, _ = encode_categoricals(test, active_cats, encoders)

    # Feature columns
    feature_cols = get_feature_cols(train)
    # Add encoded cats
    for c in active_cats:
        if c in train.columns and c not in feature_cols:
            feature_cols.append(c)

    log(f"Features: {len(feature_cols)}")

    X_train_full = train[feature_cols].fillna(-999).values
    y_train_full = train["is_fraud"].values
    X_test = test[feature_cols].fillna(-999).values
    y_test = test["is_fraud"].values

    # Internal train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_frac, random_state=args.seed, stratify=y_train_full,
    )
    log(f"Train/val split: {len(X_tr):,} / {len(X_val):,}")

    # Hyperparameter tuning (optional)
    if args.tune:
        params = tune_hyperparams(X_tr, y_tr, X_val, y_val, args.n_trials)
    else:
        params = DEFAULT_PARAMS.copy()

    # Scale pos weight
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    params["scale_pos_weight"] = round(n_neg / n_pos, 2)
    log(f"scale_pos_weight: {params['scale_pos_weight']}")

    # Train
    log("Training LightGBM ...")
    t0 = time.time()

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    elapsed = time.time() - t0
    log(f"Training complete in {elapsed:.1f}s  "
        f"| best iteration: {model.best_iteration_}")

    # Threshold optimization
    if args.threshold is not None:
        threshold = args.threshold
        log(f"Using fixed threshold: {threshold}")
    else:
        log("Optimizing decision threshold on validation set ...")
        y_val_prob = model.predict_proba(X_val)[:, 1]
        threshold, f1, prec, rec = optimize_threshold(y_val, y_val_prob, min_recall=args.min_recall)
        log(f"Optimal threshold: {threshold:.4f}  "
            f"(F1={f1:.4f}, P={prec:.4f}, R={rec:.4f})")

    # Evaluate on holdout
    log("Evaluating on test holdout ...")
    eval_str, y_prob = evaluate(model, X_test, y_test, threshold, "Test Holdout")
    print(eval_str)

    pr_auc = average_precision_score(y_test, y_prob)
    plot_pr_curve(y_test, y_prob, threshold, pr_auc, PLOTS_DIR)

    # Feature importance
    log("Computing feature importances ...")

    gain_imp = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": gain_imp,
        "split": split_imp,
        "gain_norm": gain_imp / gain_imp.sum(),
        "split_norm": split_imp / split_imp.sum(),
    }).sort_values("gain", ascending=False)


    # SHAP on a sample
    sample_idx = np.random.RandomState(args.seed).choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    X_shap = pd.DataFrame(X_test[sample_idx], columns=feature_cols)
    shap_df = compute_shap(model, X_shap, feature_cols, PLOTS_DIR)

    if shap_df is not None:
        imp_df = imp_df.merge(
            shap_df.rename(columns={"mean_abs_shap": "shap_importance"}), on="feature", how="left")

    imp_df.to_csv(IMPORTANCE_PATH, index=False)
    log(f"Feature importance saved -> {IMPORTANCE_PATH}")

    # Top 15 gain features
    print("\nTop 15 features (gain):")
    print(imp_df[["feature","gain_norm","shap_importance"]].head(15).to_string(index=False))

    # LightGBM importance plot
    plt.figure(figsize=(10, 8))
    top = imp_df.head(25)
    plt.barh(top["feature"][::-1], top["gain_norm"][::-1], color="#3498db")
    plt.xlabel("Normalized Gain")
    plt.title("Top 25 Features - LightGBM Gain Importance", fontweight="bold")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()


    # Score and save test set
    log("Scoring test holdout ...")
    test["FraudScore"] = y_prob
    test["Decision"] = pd.cut(
        test["FraudScore"],
        bins=[-1, threshold * 0.6, threshold, 1.01],
        labels=["approve", "review", "decline"],
    )
    test.to_csv(SCORED_PATH, index=False, compression="gzip")
    log(f"Scored test set -> {SCORED_PATH}")

    decision_counts = test["Decision"].value_counts()
    print("\nDecision distribution on holdout:")
    for d, n in decision_counts.items():
        fraud_in_bin = test[test["Decision"]==d]["is_fraud"].sum()
        print(f"  {d:<10}: {n:>8,}  ({n/len(test):.1%})  "
              f"| fraud={fraud_in_bin:,}  ({fraud_in_bin/max(n,1):.2%})")
        

    # Save model + metadata
    log("Saving model artifacts ...")
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders}, f)

    with open(FEAT_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": threshold, "pr_auc": pr_auc}, f, indent=2)

    with open(REPORT_PATH, "w") as f:
        f.write(eval_str)
        f.write(f"\n\nTop 20 features (gain):\n")
        f.write(imp_df[["feature","gain_norm"]].head(20).to_string(index=False))

    print("\n" + "-" * 65)
    print("[DONE] Model artifacts saved:")
    print(f"Model      : {MODEL_PATH}")
    print(f"Features   : {FEAT_PATH}")
    print(f"Threshold  : {THRESH_PATH}")
    print(f"Report     : {REPORT_PATH}")
    print(f"Plots      : {PLOTS_DIR}/")
    print(f"Scored set : {SCORED_PATH}")
    print("-" * 65)


if __name__ == "__main__":
    main()


