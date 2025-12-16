import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Load the repeated metrics
df = pd.read_csv("outputs/repeats_comparison.csv")

os.makedirs("outputs/figures", exist_ok=True)

# ---------------------------
# 1. Accuracy Comparison (Bar Plot)
# ---------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=df[['xgb_acc', 'rf_acc']])
plt.title("Accuracy Comparison: XGBoost vs Random Forest (Across Seeds)")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("outputs/figures/accuracy_comparison.png", dpi=300)
plt.close()

# ---------------------------
# 2. Macro-F1 Comparison (Bar Plot)
# ---------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=df[['xgb_f1', 'rf_f1']])
plt.title("Macro-F1 Comparison: XGBoost vs Random Forest (Across Seeds)")
plt.ylabel("Macro-F1 Score")
plt.tight_layout()
plt.savefig("outputs/figures/f1_comparison.png", dpi=300)
plt.close()

# ---------------------------
# 3. Training Time Comparison
# ---------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=df[['xgb_train_s', 'rf_train_s']])
plt.title("Training Time Comparison: XGBoost vs Random Forest")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig("outputs/figures/train_time_comparison.png", dpi=300)
plt.close()

# ---------------------------
# 4. Boxplot for Accuracy
# ---------------------------
plt.figure(figsize=(8,5))
acc_df = pd.DataFrame({
    "XGBoost": df["xgb_acc"],
    "Random Forest": df["rf_acc"]
})
sns.boxplot(data=acc_df)
plt.title("Accuracy Distribution Across Seeds")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("outputs/figures/accuracy_boxplot.png", dpi=300)
plt.close()

# ---------------------------
# 5. Boxplot for Macro-F1
# ---------------------------
plt.figure(figsize=(8,5))
f1_df = pd.DataFrame({
    "XGBoost": df["xgb_f1"],
    "Random Forest": df["rf_f1"]
})
sns.boxplot(data=f1_df)
plt.title("Macro-F1 Distribution Across Seeds")
plt.ylabel("Macro-F1 Score")
plt.tight_layout()
plt.savefig("outputs/figures/f1_boxplot.png", dpi=300)
plt.close()

# ---------------------------
# 6. Line Plot (Trend Across Seeds)
# ---------------------------
plt.figure(figsize=(10,6))
plt.plot(df["seed"], df["xgb_acc"], marker='o', label="XGB Accuracy")
plt.plot(df["seed"], df["rf_acc"], marker='o', label="RF Accuracy")
plt.title("Accuracy Trend Across Seeds")
plt.xlabel("Seed")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/accuracy_trend.png", dpi=300)
plt.close()

plt.figure(figsize=(10,6))
plt.plot(df["seed"], df["xgb_f1"], marker='o', label="XGB Macro-F1")
plt.plot(df["seed"], df["rf_f1"], marker='o', label="RF Macro-F1")
plt.title("Macro-F1 Trend Across Seeds")
plt.xlabel("Seed")
plt.ylabel("Macro-F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/f1_trend.png", dpi=300)
plt.close()

print("All comparison graphs saved inside outputs/figures/")
