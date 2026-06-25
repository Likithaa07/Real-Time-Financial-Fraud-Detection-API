# Real-Time Fraud Detection API with CatBoost

An end-to-end machine learning pipeline that detects fraudulent financial transactions and serves predictions via a REST API with real-time inference.

---

## What It Does

- Detects fraud in financial transactions in real time
- Multi-model approach — Isolation Forest, XGBoost, and CatBoost
- Handles class imbalance with SMOTE
- Serves predictions via a Flask REST API
- Includes drift detection and automated model retraining

---

## Models Used

| Model | Type |
|---|---|
| Isolation Forest | Unsupervised anomaly detection |
| One-Class SVM | Outlier detection |
| Random Forest | Supervised baseline |
| XGBoost | Gradient boosting |
| **CatBoost** | **Final production model** |

---

## Tech Stack

```
ML:       CatBoost · XGBoost · scikit-learn · imbalanced-learn
API:      Flask · pyngrok
Data:     pandas · NumPy
Viz:      Matplotlib · Seaborn
```

---

## API Usage

```bash
POST /predict
```

```json
Request:
[{"TransactionDate": "2024-06-20 15:30:00", "TransactionType": "Purchase", "Location": "Store1", "Amount": 250.00, "MerchantID": "M001"}]

Response:
{"predicted_probabilities": [0.03], "predictions": [0]}
```

`0` = Legitimate · `1` = Fraudulent

---

## Pipeline Overview

18-step pipeline covering data ingestion → preprocessing → feature engineering → model training → threshold tuning → API deployment → drift detection → automated retraining.

---

## Future Improvements

- Migrate to FastAPI for async inference
- Permanent deployment on Render or HuggingFace Spaces
- SHAP explainability per prediction
- Streamlit monitoring dashboard

---

