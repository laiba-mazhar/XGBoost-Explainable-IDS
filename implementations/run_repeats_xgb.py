# run_repeats_xgb.py
import json, time, os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from improved_ids_xgb_shap import load_and_label_data, preprocess_nsl_kdd, train_xgboost_multiclass, RANDOM_STATE

out = []
from improved_ids_xgb_shap import TRAIN_FILE, TEST_FILE
df = load_and_label_data(TRAIN_FILE, TEST_FILE, None)
X, y_series, _ = preprocess_nsl_kdd(df)
le = LabelEncoder(); y = le.fit_transform(y_series)
n_classes = len(le.classes_)
seeds = [42,7,21,99,123]

for s in seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
    t0 = time.time()
    clf, _ = train_xgboost_multiclass(X_train, y_train, n_classes, do_grid_search=False)
    tr_time = time.time() - t0
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    out.append({'seed': s, 'acc': float(accuracy_score(y_test, y_pred)), 'f1_macro': float(f1_score(y_test, y_pred, average='macro')), 'train_time_s': tr_time})
    print("seed", s, "acc", out[-1]['acc'], "f1_macro", out[-1]['f1_macro'], "t(s)", round(tr_time,2))

with open(os.path.join("outputs","xgb_repeats_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved outputs/xgb_repeats_metrics.json")
