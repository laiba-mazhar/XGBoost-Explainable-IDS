# analyze_results.py
import json, numpy as np, scipy.stats as stats, pandas as pd
x = json.load(open("outputs/xgb_repeats_metrics.json"))
r = json.load(open("outputs/rf_repeats_metrics.json"))

x_acc = np.array([m['acc'] for m in x])
r_acc = np.array([m['acc'] for m in r])
x_f1 = np.array([m['f1_macro'] for m in x])
r_f1 = np.array([m['f1_macro'] for m in r])

def stats_str(a):
    return f"{a.mean():.4f} ± {a.std():.4f}"

print("Accuracy XGB:", stats_str(x_acc))
print("Accuracy RF :", stats_str(r_acc))
print("Macro-F1 XGB :", stats_str(x_f1))
print("Macro-F1 RF  :", stats_str(r_f1))

t_acc, p_acc = stats.ttest_rel(x_acc, r_acc)
t_f1, p_f1 = stats.ttest_rel(x_f1, r_f1)
print("paired t-test acc p:", p_acc)
print("paired t-test f1  p:", p_f1)

# Save a comparison CSV
df = pd.DataFrame({
    'seed':[m['seed'] for m in x],
    'xgb_acc':[m['acc'] for m in x],
    'rf_acc':[m['acc'] for m in r],
    'xgb_f1':[m['f1_macro'] for m in x],
    'rf_f1':[m['f1_macro'] for m in r],
    'xgb_train_s':[m['train_time_s'] for m in x],
    'rf_train_s':[m['train_time_s'] for m in r],
})
df.to_csv("outputs/repeats_comparison.csv", index=False)
print("Saved outputs/repeats_comparison.csv")
