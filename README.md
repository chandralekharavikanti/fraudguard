# FraudGuard AI — Real Credit Card Fraud Detection

End-to-end machine learning on the **real anonymized ULB/Worldline credit-card transaction dataset** (September 2013). No synthetic data fallback.

## Dataset
Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at `data/raw/creditcard.csv`. The raw dataset is excluded from Git. It has 284,807 transactions, 492 fraud labels, and features `Time`, `V1`–`V28`, `Amount`, `Class`.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
python -m fraudguard.eda --data data/raw/creditcard.csv
python -m fraudguard.train --data data/raw/creditcard.csv --split temporal
pytest -q
```

Training compares class-weighted Logistic Regression and Random Forest. Preprocessing is fitted only on training data. Default is an 80/20 time-ordered holdout after deduplication. Metrics include PR-AUC (average precision), precision, recall, F1, and confusion matrix. The 0.5 decision threshold is a baseline, not a production threshold.

## Measured real-data baseline (2026-10-20)
After removing 1,081 duplicate rows, the time-ordered holdout had 226,980 train and 56,746 test transactions (74 frauds in test). Logistic Regression: **PR-AUC 0.7611; precision 0.0661; recall 0.8919; F1 0.1230**. Confusion matrix: `[[55739,933],[8,66]]`. See `reports/real_data_baseline.json`.

## Limitations
Historical anonymized data is not representative of live production fraud and PCA components cannot be interpreted as merchant/customer attributes. Next: validation-based threshold tuning, XGBoost, SHAP, FastAPI, and dashboard.
