# run_baseline_rf.py
import time, joblib, json, os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# import loader and preprocess helpers and the TRAIN_FILE/TEST_FILE constants
from improved_ids_xgb_shap import load_and_label_data, preprocess_nsl_kdd, TRAIN_FILE, TEST_FILE, RANDOM_STATE, OUTPUT_DIR

# load data (pass TRAIN_FILE and TEST_FILE explicitly)
df = load_and_label_data(TRAIN_FILE, TEST_FILE, None)
X, y_series, scaler = preprocess_nsl_kdd(df)
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y_series)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=RANDOM_STATE, stratify=y_enc)

# train RF
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
t0 = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - t0

# eval
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
with open(os.path.join(OUTPUT_DIR, "baseline_rf_classification_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("RF train time (s):", train_time)
print(classification_report(y_test, y_pred))
joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_baseline_model.joblib"))
