# Credit-Card Fraud Detection

An intermediate binary-classification project on a **heavily imbalanced**
dataset: only 0.17% of transactions are fraud. It trains a class-weighted
logistic-regression model and evaluates it with the metrics that actually matter
when one class is 600× rarer than the other.

This project is educational only. It is not a production fraud system.

## Dataset

Download Kaggle's [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset and place `creditcard.csv` at:

```text
data/raw/creditcard.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

The file is ~66 MB, 284,807 rows, 31 columns. Features `V1`–`V28` are
already-anonymized PCA components; only `Time` and `Amount` are raw. `Class` is
the target (1 = fraud). After removing 1,081 exact duplicate rows, 283,726 remain.

## The imbalance problem

With 0.17% fraud, a model that predicts **"never fraud"** scores 99.83% accuracy
while catching zero fraud. So accuracy is deliberately **not** reported. Instead:

- **Recall** — of all real fraud, how much did we catch?
- **Precision** — of everything we flagged, how much was really fraud?
- **PR-AUC** (average precision) — the honest threshold-free score for rare
  positives. Prefer it to ROC-AUC here, which looks flattering on imbalanced data.

`class_weight="balanced"` re-weights the rare class so the model is penalized for
ignoring fraud rather than rewarded for always predicting "legitimate."

## Workflow

1. Load the raw CSV and remove exact duplicate rows.
2. Scale `Time` and `Amount`; pass the PCA components through unchanged.
3. Make a **stratified** 80/20 split (so both folds keep the fraud ratio).
4. Train class-weighted logistic regression in a leakage-safe pipeline.
5. Evaluate with precision, recall, F1, ROC-AUC, and PR-AUC.
6. Save a precision-recall curve.

## Results

| Measurement | Result |
|---|---:|
| Recall (fraud caught) | 0.87 |
| Precision (flags that were real) | 0.06 |
| ROC-AUC | 0.97 |
| **PR-AUC (average precision)** | **0.67** |

The class weighting buys high recall — the model catches ~87% of fraud — at the
cost of low precision: most flagged transactions are false alarms. That is the
central trade-off in fraud detection, and tuning the decision threshold along the
precision-recall curve is the natural next experiment. Note how ROC-AUC (0.97)
looks great while PR-AUC (0.67) tells the more honest story on imbalanced data.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The precision-recall curve is written to
`reports/figures/precision_recall_curve.png`. Training on 284k rows takes a few
seconds.

## Layout

```text
data/raw/         downloaded dataset (git-ignored — see Dataset above)
src/preprocess.py loading, cleaning, scaling
src/train.py      pipeline, training loop, evaluation, figure
src/evaluate.py   imbalance-aware metric helpers
main.py           entry point
```

## Why this project

Inspired by classmate fraud-detection projects (e.g. Srijansarkar17's
`Credit-Card-Fraud-Detection-with-Fast-API`, adityaxdubey's
`Payment-Fraud-Detection`). It is the intermediate step up from the balanced
classifiers in the sibling `ml-*` folders — same pipeline shape, but the metrics
and class weighting change to handle severe imbalance.
