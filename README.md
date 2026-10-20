# FraudGuard — Real Credit Card Fraud Detection

**Machine Learning / Data Science portfolio project** | Real ULB/Worldline credit-card transactions | Python, scikit-learn, pandas

## Project goal
Identify rare fraudulent credit-card transactions while measuring the operational cost of false alerts. This is a reproducible model-evaluation project, **not** a dashboard or a production fraud-screening service.

## Dataset
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 real anonymized transactions, of which 492 (0.173%) are fraud.
- Inputs: `Time`, `Amount`, and anonymized PCA features `V1`–`V28`. Target: `Class` (1 = fraud).
- Historical September 2013 data; no synthetic data is used.
- The CSV is intentionally not committed. Download it and place it in `data/raw/creditcard.csv`.

## Reproduce the analysis
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
python -m fraudguard.eda --data data/raw/creditcard.csv
python -m fraudguard.train --data data/raw/creditcard.csv --split temporal
pytest -q
```

EDA output: `reports/summary.json` and `reports/class_distribution.png`. Model comparison: `artifacts/metrics.json` (generated locally; ignored by Git). Model binaries: `artifacts/*.joblib`.

## Methodology
1. Validate real-data schema and missing values.
2. Remove 1,081 exact duplicate records before splitting.
3. Use an 80/20 **time-ordered** holdout: 226,980 training and 56,746 test transactions, with 74 fraud cases in the test set.
4. Compare class-weighted Logistic Regression and Random Forest; preprocessing is fitted only on the training partition.
5. Evaluate **PR-AUC (average precision), precision, recall, F1, and confusion matrix** rather than accuracy alone.

## Verified baseline: Logistic Regression
At the default 0.5 classification threshold on the held-out test set:

| Metric | Result |
| --- | ---: |
| PR-AUC / average precision | **0.7611** |
| Precision | **6.61%** |
| Recall | **89.19%** |
| F1 | **12.30%** |
| True positives (frauds detected) | **66** |
| False negatives (frauds missed) | **8** |
| False positives (legitimate transactions flagged) | **933** |
| True negatives | **55,739** |

See [the saved baseline results](reports/real_data_baseline.json).

### Interpretation
The baseline detects most fraud cases but generates many false alarms: roughly **14 false alerts for each correctly detected fraud** (933 / 66). In a real review queue, this would impose significant operational cost. The next evaluation question is whether Random Forest achieves a better precision–recall balance and whether a threshold selected on **validation data** can reduce false positives without sacrificing too much recall.

### Model comparison: Logistic Regression vs Random Forest

Both models were evaluated on the same **56,746-transaction time-ordered holdout** (74 fraudulent transactions) at the default **0.5 classification threshold**.

| Metric | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| PR-AUC / average precision | 0.7611 | **0.8127** |
| Precision | 6.61% | **98.18%** |
| Recall | **89.19%** | 72.97% |
| F1-score | 12.30% | **83.72%** |
| True positives (frauds caught) | **66** | 54 |
| False negatives (frauds missed) | **8** | 20 |
| False positives (false alarms) | 933 | **1** |
| True negatives | 55,739 | **56,671** |

**Conclusion:** At the evaluated threshold, **Random Forest is the stronger operational starting point** for a low-false-alarm fraud review queue: its PR-AUC is higher, and it produces just one false alert versus 933 for Logistic Regression. The trade-off is material: Random Forest misses **20 fraud cases**, compared with **8** for Logistic Regression. Neither model can be declared universally best without business-specific costs for missed fraud and investigation workload.

**Important evaluation caveat:** These numbers come from one historical test holdout, not a live production evaluation. Threshold tuning or model selection should be performed on a separate validation partition, followed by a final untouched test evaluation. The test comparison here is descriptive, not a claim of unbiased optimized performance.

## Limitations and responsible interpretation
- This is historical, anonymized benchmark data, **not a production-ready fraud detector**.
- PCA features cannot support explanations about merchant, cardholder, or purchase behavior.
- Test-set results should not be used to tune thresholds. A separate validation split is needed for model/threshold selection.
- Recall alone is not enough; precision and false-positive volume matter to fraud operations.
- This repository deliberately focuses on ML methodology, not a Streamlit dashboard.

## Portfolio takeaway
This project demonstrates **real-data validation, class imbalance handling, time-aware evaluation, reproducible sklearn pipelines, and critical interpretation of fraud detection trade-offs**.
