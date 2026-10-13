# FraudShield AI — Intelligent Financial Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **⚠️ Educational Prototype**: This application is built for a postgraduate Machine Learning
> Laboratory CAT assessment. It is NOT suitable for production financial decision-making without
> further security validation, bias analysis, and regulatory compliance review.

---

## Overview

FraudShield AI is an end-to-end financial fraud detection system that demonstrates:

- Real-world ML pipeline on the European Credit Card Fraud dataset
- Supervised classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Unsupervised anomaly detection: Isolation Forest
- Class imbalance handling: SMOTE (training only)
- Explainable AI: SHAP values for global and local explanations
- REST API backend: Flask
- Professional analytics dashboard: HTML5 / Chart.js

---

## Objectives

1. Identify fraudulent transactions in highly imbalanced financial data
2. Compare supervised and unsupervised ML approaches
3. Explain model decisions using SHAP
4. Demonstrate Precision/Recall trade-offs for business stakeholders
5. Build a full-stack application suitable for academic assessment

---

## Dataset

**Source**: European Credit Card Fraud Dataset (2013)
- **Official download**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Alternative**: https://zenodo.org/records/7395559
- **Citation**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
  *Calibrating Probability with Undersampling for Unbalanced Classification.*
  2015 IEEE Symposium Series on Computational Intelligence (SSCI), Cape Town, South Africa.

**Dataset details**:
- 284,807 transactions (September 2013, European cardholders)
- 492 fraud cases (0.172%)
- Features: V1–V28 (PCA-transformed for privacy), Time (seconds), Amount, Class (0/1)
- This dataset is redistributed under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)

**⚠️ Important**: Do NOT commit `creditcard.csv` to version control.

---

## Installation

### Prerequisites
- Python 3.11+
- pip

### Steps

```powershell
# 1. Clone / open project
cd C:\Users\Hp\Documents\Fraud-Detection-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it at: `data/creditcard.csv`
3. Verify with: `python -c "import pandas as pd; df = pd.read_csv('data/creditcard.csv'); print(df.shape)"`
   — Expected: `(284807, 31)`

---

## Training the Models

```powershell
# Option A: Full training on official European Credit Card dataset (284,807 rows)
python train_pipeline.py

# Option B: Fast experimentation mode (stratified sample — labelled as SAMPLED)
python train_pipeline.py --sample 50000

# Option C: Demonstration mode using isolated synthetic demo data
python train_pipeline.py --demo
```

Artifacts are saved to:
- `models/` — trained model `.joblib` files + scaler
- `results/reports/` — model comparison JSON, EDA cache
- `results/cases.db` — SQLite database for fraud investigation cases
- `models/dataset_metadata.json` — training metadata

---

## Running the Application

```powershell
# Start Flask backend (in one terminal)
python backend/app.py

# Open in browser
start http://localhost:5000

# Or use the convenience script
run.bat
```

---

## Running Tests

```powershell
# Run all tests
pytest tests/ -v

# Run only fast unit tests (no API)
pytest tests/test_data_loader.py tests/test_preprocessing.py tests/test_models.py -v

# Run API tests
pytest tests/test_api.py -v

# With coverage (requires pytest-cov)
pytest tests/ --cov=src --cov=backend --cov-report=term-missing
```

---

## Project Structure

```
Fraud-Detection-AI/
├── backend/            Flask application
│   ├── app.py          Entry point
│   ├── routes.py       All REST API endpoints
│   └── services.py     Lazy-loading data/model services
├── src/                Core ML pipeline
│   ├── config.py       All configuration
│   ├── data_loader.py  Dataset loading & validation
│   ├── preprocessing.py Scaling, split, SMOTE
│   ├── eda.py          Exploratory data analysis
│   ├── train_models.py Model factories & persistence
│   ├── evaluate_models.py Metrics & comparison report
│   ├── anomaly_detection.py Isolation Forest scoring
│   ├── explainability.py SHAP integration
│   ├── business_insights.py Derived business conclusions
│   └── utils.py        Shared helpers
├── frontend/           Dashboard UI
│   ├── index.html      Single-page app
│   ├── css/style.css   Dark analytics theme
│   └── js/             app.js, api.js, charts.js
├── data/               Place creditcard.csv here
├── models/             Trained model artifacts
├── results/            Reports and figures
│   ├── reports/        JSON evaluation reports
│   └── figures/        Generated plots
├── tests/              Automated test suite
├── docs/               Academic documentation
├── train_pipeline.py   Main training script
├── requirements.txt    Python dependencies
├── run.bat             Windows quick-start
└── README.md
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: creditcard.csv` | Download from Kaggle and place in `data/` |
| `Cannot reach backend` | Run `python backend/app.py` in a terminal |
| `No model artifacts found` | Run `python train_pipeline.py` after loading data |
| `SHAP unavailable` | Ensure `shap==0.45.1` is installed; check terminal for traceback |
| `XGBoost warning about eval_metric` | Safe to ignore — uses `logloss` by default |
| Slow training | Use `python train_pipeline.py --sample 50000` |
| Import errors | Ensure you are in the project root when running scripts |

---

## Limitations

- Trained on a 2013 dataset; real fraud patterns evolve rapidly
- PCA-transformed features (V1–V28) are not interpretable in isolation
- SMOTE can create synthetic minority samples that don't reflect real fraud
- Model evaluation is on a held-out test set — real-world performance may differ
- This is an educational prototype; NOT for production without further review

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Backend health status |
| `/api/dataset/summary` | GET | Dataset statistics |
| `/api/eda` | GET | Full EDA data |
| `/api/eda/feature?name=V4` | GET | Single feature distribution |
| `/api/models/comparison` | GET | Full model comparison report |
| `/api/models/status` | GET | Model artifact availability |
| `/api/predict` | POST | Single transaction prediction |
| `/api/explain` | POST | SHAP local explanation |
| `/api/explain/global` | GET | SHAP global importance |
| `/api/anomaly` | GET | Isolation Forest anomaly scores |
| `/api/business-insights` | GET | Business intelligence from results |
| `/api/transactions` | GET | Paginated transaction list |
| `/api/threshold-analysis` | GET | Precision/Recall at thresholds |
| `/api/cases` | GET | List investigation cases with filters |
| `/api/cases` | POST | Create an investigation case |
| `/api/cases/<id>` | PATCH | Update case status or notes |
| `/api/cases/<id>` | DELETE | Delete an investigation case |
