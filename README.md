# Credit Card Fraud Detection

An imbalanced classification pipeline for detecting fraudulent credit card transactions, built as a group project for TTTC2453 (Machine Learning). The pipeline moves from a linear baseline through an ensemble model to a deep learning model, comparing each on metrics suited to extreme class imbalance rather than raw accuracy.

## Problem

The dataset (a standard Kaggle credit card fraud dataset, ~284,800 transactions) is severely imbalanced: only **0.17%** of transactions are fraudulent. A model that predicts "legit" for everything would score 99.83% accuracy while catching zero fraud — so accuracy is discarded as a metric in favor of **recall**, **precision**, and **AUC-PR** (area under the precision-recall curve), which reflect how well the model actually finds fraud without drowning in false alarms.

## Pipeline

1. **Stage 0 — Data audit, cleaning & EDA**: missing-value and duplicate checks, class distribution, outlier analysis, feature correlation, log-transform of transaction amount.
2. **Stage 0b — Imbalance handling**: compared SMOTE (synthetic oversampling) against class weighting; class weighting was chosen as it improved minority-class focus without introducing synthetic samples.
3. **Stage A — Baseline model**: Logistic Regression with class weighting.
4. **Stage B — Ensemble model**: XGBoost, tuned via randomized search over a 5-fold cross-validated grid.
5. **Stage C — Deep learning model**: a Multi-Layer Perceptron (Keras/TensorFlow) with early stopping.
6. **Final comparison**: all three models evaluated on a held-out test set using AUC-PR as the primary metric.

## Results

Held-out test set, default 0.50 classification threshold:

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC | AUC-PR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9750 | 0.0558 | 0.8737 | 0.1049 | 0.9636 | 0.6807 |
| **XGBoost** | **0.9992** | **0.7576** | **0.7895** | **0.7732** | **0.9674** | **0.8093** |
| MLP | 0.9794 | 0.0661 | 0.8632 | 0.1228 | 0.9608 | 0.6614 |

**XGBoost was the best-performing model** (AUC-PR 0.8093), catching ~79% of fraud while keeping ~76% of its fraud alerts correct. Logistic Regression and the MLP both reach higher recall but at a steep precision cost (over 90% of their "fraud" flags are false alarms), making them impractical to act on directly without further threshold tuning.

Feature importance analysis showed `V14`, `V12`, and `V10` (PCA-derived features from the original dataset) as the strongest predictors of fraud across models.

## Repository structure

```
├── notebooks/
│   └── Credit_Card_Fraud.ipynb   # full pipeline: EDA → modeling → evaluation
├── reports/
│   ├── Credit Card Fraud Report.pdf
│   └── Credit Card Fraud Poster.pdf
├── requirements.txt
└── README.md
```

## Running it yourself

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook notebooks/Credit_Card_Fraud.ipynb
```

The notebook downloads the dataset directly from a public URL at runtime, so no manual data download is required.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, TensorFlow/Keras, matplotlib, seaborn.

## Team

Group project for TTTC2453 — Machine Learning.
