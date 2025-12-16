# Explainable Multiclass Intrusion Detection Using XGBoost and SHAP

This repository contains the implementation and research artifacts for a robust and explainable
multiclass Intrusion Detection System (IDS) based on XGBoost, TreeSHAP explainability, and
multi-seed statistical validation.

The proposed framework is designed for Security Operations Center (SOC) environments,
providing transparent alert explanations and severity-aware risk prioritization.

---

## Project Overview

The system performs multiclass intrusion detection on network traffic and enhances
traditional IDS pipelines by integrating:

- XGBoost-based multiclass classification
- Multi-seed evaluation for robustness
- Per-sample explainability using TreeSHAP
- Severity scoring for SOC-oriented alert triage
- Cross-dataset validation (NSL-KDD and UNSW-NB15)

---

## Repository Structure

```text
├── implementation/
│   ├── analyze_results.py
│   ├── improved_ids_xgb_shap.py
│   ├── plot_comparison_figures.py
│   ├── run_baseline_rf.py
│   ├── run_repeats_rf.py
│   └── run_repeats_xgb.py
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── figures/
        ├── macro_f1_across_seeds.png
        └── severity_distribution.png
````

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run XGBoost Experiments

```bash
python implementation/run_repeats_xgb.py
```

### 3. Run Random Forest Baseline

```bash
python implementation/run_repeats_rf.py
```

### 4. Analyze Results

```bash
python implementation/analyze_results.py
python implementation/plot_comparison_figures.py
```

---

## Datasets

Experiments were conducted on:

* NSL-KDD
* UNSW-NB15

Datasets are not included due to size and licensing restrictions.
Please download them from official sources and update file paths accordingly.

---

## Explainability and Severity Scoring

TreeSHAP is used to generate per-sample explanations for each alert.
A severity score is computed to prioritize alerts into operational risk levels:
Low, Medium, High, and Critical.

This enables actionable and evidence-backed SOC decision-making.

---

## Authors

* **Laiba Mazhar**
  FAST National University of Computing and Emerging Sciences, Islamabad
  Email: [i221855@nu.edu.pk](mailto:i221855@nu.edu.pk)

* **Abyaz Israr**
  FAST National University of Computing and Emerging Sciences, Islamabad
  Email: [i222056@nu.edu.pk](mailto:i222056@nu.edu.pk)

* **Tashfeen Abbasi**
  FAST National University of Computing and Emerging Sciences, Islamabad
  Email: [i222041@nu.edu.pk](mailto:i222041@nu.edu.pk)

---

## License

This repository is intended for academic and research use.
Please cite the associated paper if you use this work.
