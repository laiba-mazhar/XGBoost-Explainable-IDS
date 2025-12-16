#!/usr/bin/env python3
"""
improved_ids_xgb_shap_unsw_final_fixed.py

Final cleaned script:
- 6-class mapping (UNSW -> DoS-Fuzzers, Exploits, Generic, Normal, RareAttacks, Reconnaissance)
- XGBoost multiclass with sample weights
- Stable SHAP using shap.Explainer + predict_proba callable (Option B)
- Forces numeric float64 arrays for model & SHAP
- Multi-seed evaluation, reporting, plots, severity scoring, saved outputs

Author: Assistant (final)
"""

import os
import json
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
import shap

# ----------------------------
# Configuration
# ----------------------------
# Update DATA_FILE if your CSV is in a different folder
DATA_FILE = r"C:/Users/laiba/Desktop/CNET EXPERIMENTS/data/UNSW_NB15_training-set.csv"
OUTPUT_DIR = os.path.join(os.getcwd(), "unsw_outputs_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEEDS = [42, 7, 21, 99, 123]
TEST_SIZE = 0.25

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", message=r".*use_label_encoder.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Helpers & mapping
# ----------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_unsw(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
    elif 'attack' in df.columns:
        df['attack_cat'] = df['attack'].astype(str).str.strip()
    elif 'label' in df.columns and df['label'].dtype != object:
        df['attack_cat'] = df['label'].apply(lambda x: 'Normal' if int(x) == 0 else 'Attack')
    else:
        df['attack_cat'] = df.iloc[:, -1].astype(str).str.strip()
    df['attack_cat'] = df['attack_cat'].replace({'BENIGN': 'Normal', 'normal': 'Normal'})
    return df

def map_to_6classes(cat):
    s = str(cat).lower()
    if s in ['normal', 'benign']:
        return 'Normal'
    if 'generic' in s:
        return 'Generic'
    if 'exploit' in s:
        return 'Exploits'
    if 'recon' in s or 'scan' in s or 'reconnaissance' in s:
        return 'Reconnaissance'
    if 'dos' in s or 'ddos' in s or 'fuzz' in s:
        return 'DoS-Fuzzers'
    if any(k in s for k in ['backdoor', 'shellcode', 'analysis', 'worm']):
        return 'RareAttacks'
    return 'RareAttacks'

def preprocess_and_encode(df):
    df = df.copy()
    df['attack_family_6'] = df['attack_cat'].apply(map_to_6classes)

    # drop common non-feature columns if present
    for c in ['id', 'session_id', 'timestamp', 'ts', 'attack', 'label']:
        if c in df.columns and c not in ['attack_cat', 'attack_family_6']:
            try:
                df.drop(columns=c, inplace=True)
            except Exception:
                pass

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ['attack_cat', 'attack_family_6']]

    numeric_cols = [c for c in df.columns if c not in categorical_cols + ['attack_cat', 'attack_family_6']]

    # coerce numeric cols to numeric, fillna
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # one-hot encode categorical
    df_cat = pd.get_dummies(df[categorical_cols].astype(str), prefix=categorical_cols, drop_first=False) if categorical_cols else pd.DataFrame(index=df.index)

    # scale numeric
    scaler = StandardScaler()
    if numeric_cols:
        df_num = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    else:
        df_num = pd.DataFrame(index=df.index)

    X = pd.concat([df_num, df_cat], axis=1)
    # enforce numeric float64
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)

    y = df['attack_family_6']

    return X, y, scaler

# ----------------------------
# Training functions
# ----------------------------
def train_xgb_with_weights(X_train, y_train, n_classes, seed=None):
    # compute weights on unique labels (encoded ints)
    classes, counts = np.unique(y_train, return_counts=True)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw_map = {c: w for c, w in zip(classes, cw)}
    sample_weights = np.array([cw_map[int(v)] for v in y_train])

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=350,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.5,
        tree_method='hist',
        random_state=seed,
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

# ----------------------------
# Stable SHAP (Option B) - safe for multiclass XGBoost
# ----------------------------
def compute_tree_shap_safe(clf, X_ref_df, X_explain_df, out_dir, sample_for_local=0):
    ensure_dir(out_dir)

    X_ref = X_ref_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
    X_explain = X_explain_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)

    X_ref_arr = X_ref.to_numpy(dtype=np.float64)
    X_explain_arr = X_explain.to_numpy(dtype=np.float64)

    feature_names = X_explain.columns.tolist()
    n_classes = clf.n_classes_
    shap_vals_per_class = {}

    # Loop over each class
    for cls in range(n_classes):

        def predict_single_class(X):
            return clf.predict_proba(X)[:, cls]

        masker = shap.maskers.Independent(X_ref_arr, max_samples=100)

        explainer = shap.Explainer(
            predict_single_class,
            masker,
            model_output="raw",      # MUST be raw for single-output models
            algorithm="auto"
        )

        shap_values = explainer(X_explain_arr)  # (n_samples, n_features)
        vals = getattr(shap_values, "values", shap_values)

        if vals.ndim != 2 or vals.shape[1] != X_explain.shape[1]:
            raise ValueError(
                f"SHAP for class {cls} returned shape {vals.shape}, expected (n_samples, n_features)"
            )

        shap_vals_per_class[cls] = vals

        # Save global importance plot
        mean_abs = np.mean(np.abs(vals), axis=0)
        feat_imp = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=feat_imp.values[:30], y=feat_imp.index[:30])
        plt.title(f"Class {cls} mean |SHAP| - Top features")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_mean_abs_class{cls}.png"), dpi=200)
        plt.close()
        
    return shap_vals_per_class, None


# ----------------------------
# Severity scoring
# ----------------------------
def compute_severity_scores(pred_labels, pred_proba, X_df, shap_values_per_class, label_encoder):
    base_map = {
        "Normal": 0.0,
        "Generic": 0.5,
        "Exploits": 0.9,
        "Reconnaissance": 0.6,
        "DoS-Fuzzers": 0.95,
        "RareAttacks": 1.0
    }
    rows = []
    X_df = X_df.reset_index(drop=True)
    n = X_df.shape[0]
    features = X_df.columns.tolist()

    for i in range(n):
        cls_idx = int(pred_labels[i])
        try:
            cls_name = label_encoder.inverse_transform([cls_idx])[0]
        except Exception:
            cls_name = str(cls_idx)
        conf = float(np.max(pred_proba[i]))
        base = float(base_map.get(cls_name, 0.5))
        impact = 0.0
        top_feats = []
        try:
            if isinstance(shap_values_per_class, dict) and cls_idx in shap_values_per_class:
                sv = shap_values_per_class[cls_idx]
                if sv.shape[0] == n:
                    row_sv = sv[i]
                else:
                    row_sv = sv[i] if i < sv.shape[0] else sv[0]
                s = pd.Series(np.abs(row_sv), index=features)
                top = s.sort_values(ascending=False).head(6)
                impact = float(np.mean(top.values))
                top_feats = list(zip(top.index.tolist(), top.values.tolist()))
        except Exception:
            pass
        severity_score = base * conf + impact
        rows.append({
            "sample_index": i,
            "predicted_class_index": cls_idx,
            "predicted_class": cls_name,
            "confidence": conf,
            "base_severity": base,
            "impact_modifier": impact,
            "severity_score": severity_score,
            "top_shap_features": top_feats
        })
    return pd.DataFrame(rows)

# ----------------------------
# Evaluation & reporting
# ----------------------------
def evaluate_and_save(clf, X_test, y_test_enc, label_encoder, shap_vals_per_class, out_dir):
    ensure_dir(out_dir)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    target_names = list(label_encoder.classes_)
    rep = classification_report(y_test_enc, y_pred, target_names=target_names, output_dict=True)
    with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print(classification_report(y_test_enc, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    alerts_df = compute_severity_scores(y_pred, y_proba, X_test.reset_index(drop=True), shap_vals_per_class, label_encoder)
    alerts_df.to_csv(os.path.join(out_dir, "alerts_with_severity.csv"), index=False)

    preds_df = X_test.reset_index(drop=True).copy()
    preds_df['predicted_class'] = [label_encoder.inverse_transform([int(p)])[0] for p in y_pred]
    preds_df['pred_conf'] = [float(np.max(p)) for p in y_proba]
    preds_df.to_csv(os.path.join(out_dir, "predictions_summary.csv"), index=False)

    return rep, cm, alerts_df

# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading dataset:", DATA_FILE)
    df = load_unsw(DATA_FILE)
    print("Raw shape:", df.shape)
    print("Unique attack categories sample:", pd.Series(df['attack_cat'].unique()).tolist()[:20])

    X, y, scaler = preprocess_and_encode(df)
    print("Preprocessed X shape:", X.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("6-class mapping:", list(le.classes_))

    seed_metrics = []
    all_alerts = []
    macro_f1s = []

    for seed in RANDOM_SEEDS:
        print(f"--- Seed {seed} ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=seed)
        print("Train/Test shapes:", X_train.shape, X_test.shape)

        # enforce numeric dtype
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
        X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)

        t0 = time.time()
        clf = train_xgb_with_weights(X_train, y_train, n_classes=len(le.classes_), seed=seed)
        train_time = time.time() - t0
        joblib.dump(clf, os.path.join(OUTPUT_DIR, f"xgb_6class_seed{seed}.joblib"))

        # SHAP: pick reference from train (use up to 2000 rows) and explain subset from test
        ref_n = min(2000, X_train.shape[0])
        try:
            X_ref = X_train.sample(n=ref_n, random_state=seed)
        except Exception:
            X_ref = X_train.iloc[:ref_n]

        explain_n = min(500, X_test.shape[0])
        explain_idx = np.random.RandomState(seed).choice(X_test.shape[0], explain_n, replace=False)
        X_explain = X_test.iloc[explain_idx]

        shap_out_dir = os.path.join(OUTPUT_DIR, f"shap_seed{seed}")
        shap_vals_per_class, explainer = compute_tree_shap_safe(clf, X_ref, X_explain, shap_out_dir, sample_for_local=0)

        # Evaluate on full X_test
        report_dir = os.path.join(OUTPUT_DIR, f"reports_seed{seed}")
        rep, cm, alerts_df = evaluate_and_save(clf, X_test, y_test, le, shap_vals_per_class, report_dir)

        # record metrics
        macro_f1 = rep.get("macro avg", {}).get("f1-score", None)
        seed_metrics.append({"seed": seed, "train_time_s": train_time, "macro_f1": macro_f1, "report": rep})
        if macro_f1 is not None:
            macro_f1s.append(macro_f1)

        alerts_df['seed'] = seed
        all_alerts.append(alerts_df)

    # save aggregated metrics and alerts
    with open(os.path.join(OUTPUT_DIR, "aggregated_metrics.json"), "w") as f:
        json.dump(seed_metrics, f, indent=2)
    if all_alerts:
        pd.concat(all_alerts, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "all_seeds_alerts.csv"), index=False)

    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder_6class.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_6class.joblib"))

    # aggregated plots
    try:
        seeds = [m['seed'] for m in seed_metrics]
        macro_vals = [m['macro_f1'] for m in seed_metrics]
        plt.figure(figsize=(8,5))
        sns.barplot(x=[str(s) for s in seeds], y=macro_vals)
        plt.ylabel("Macro-F1")
        plt.xlabel("Seed")
        plt.title("Macro-F1 across seeds (6-class mapping)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "macro_f1_across_seeds.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    # severity distribution plot (all alerts)
    try:
        all_alerts_df = pd.concat(all_alerts, ignore_index=True)
        plt.figure(figsize=(8,5))
        sns.histplot(all_alerts_df['severity_score'], bins=30, kde=False)
        plt.title("Severity score distribution (all seeds)")
        plt.xlabel("Severity score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "severity_distribution.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    print("Done. Outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
