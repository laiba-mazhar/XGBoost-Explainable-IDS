# run_repeats_rf.py
import json, time, os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from improved_ids_xgb_shap import load_and_label_data, preprocess_nsl_kdd, TRAIN_FILE, TEST_FILE

out = []
df = load_and_label_data(TRAIN_FILE, TEST_FILE, None)
X, y_series, _ = preprocess_nsl_kdd(df)
le = LabelEncoder(); y = le.fit_transform(y_series)
seeds = [42,7,21,99,123]

for s in seeds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=s, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s)
    t0 = time.time()
    clf.fit(X_train, y_train)
    tr_time = time.time() - t0
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    out.append({'seed': s, 'acc': float(accuracy_score(y_test, y_pred)), 'f1_macro': float(f1_score(y_test, y_pred, average='macro')), 'train_time_s': tr_time})
    print("seed", s, "acc", out[-1]['acc'], "f1_macro", out[-1]['f1_macro'], "t(s)", round(tr_time,2))

with open(os.path.join("outputs","rf_repeats_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved outputs/rf_repeats_metrics.json")
