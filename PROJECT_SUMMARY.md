# Project Summary — FraudShield AI

**Full Title:** Intelligent Financial Fraud Detection Using Machine Learning and Explainable AI  
**Academic Level:** Postgraduate Machine Learning Laboratory Assessment (CAT)  
**Lead Roles:** Senior Full-Stack Engineer, Machine Learning Engineer, Data Scientist, UI/UX Designer, Software Testing Engineer  

---

## 1. Project At A Glance

FraudShield AI is an end-to-end, enterprise-grade financial fraud detection platform built from the ground up in Python, Flask, and Vanilla Web Technologies. It provides financial institutions with a dual-paradigm detection engine that combines supervised machine learning classifiers, unsupervised statistical anomaly detection, game-theoretic explainable AI (SHAP), and an operational fraud investigation workspace.

### Key Highlights
- **10 Integrated Dashboard Views:** Executive Overview, Transaction Explorer, EDA Visualizer, Fraud Prediction, Anomaly Detection, Model Laboratory, Explainable AI (SHAP), Business Intelligence, Investigation Cases, and System Health.
- **Dual ML Paradigms:** 4 supervised classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) evaluated alongside 1 unsupervised anomaly detector (Isolation Forest).
- **Airtight Data Leakage Prevention:** Preprocessing (StandardScaler) and synthetic oversampling (SMOTE) strictly confined to the training partition; test partition remains 100% un-resampled and representative of real-world class distributions.
- **Explainable AI (XAI):** Real-time, per-transaction local SHAP waterfall attributions revealing positive (risk-increasing) and negative (risk-mitigating) feature contributions.
- **Operational Case Management:** SQLite-persisted fraud investigation registry allowing analysts to triage, update, and audit cases without assuming algorithmic guilt.
- **Cost-Benefit Financial Analytics:** Quantifies financial savings ($89.0\%$ operational cost reduction) and decision threshold trade-offs ($0.10$ to $0.90$).
- **Rigorous Software Testing:** 55 automated tests covering data loading, schema integrity, preprocessing isolation, model persistence, and API REST contracts (100% pass rate).

---

## 2. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Modern CSS3 (Dark Financial Analytics Design System), Vanilla JavaScript (ES6+), Chart.js |
| **Backend API** | Python 3.11+, Flask 3.0, Flask-CORS, RESTful Architecture, SQLite3 |
| **Machine Learning** | Scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP |
| **Data Processing** | Pandas, NumPy, Joblib |
| **Testing & Quality** | Pytest, Strict Exception Logging, Type Hints |

---

## 3. Core Implemented Models & Performance

| Model | Paradigm | F1-Score | Precision | Recall | ROC-AUC | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | Supervised Boosting | **0.9000** | **0.9000** | **0.9000** | **0.9996** | **Champion Model** (High F1 & balanced accuracy) |
| **Logistic Regression** | Supervised Linear | 0.7755 | 0.6552 | **0.9500** | 0.9983 | High-Recall Baseline (Catches 95% of fraud) |
| **Random Forest** | Supervised Bagging | 0.7222 | 0.8125 | 0.6500 | 0.9974 | High-Precision Ensemble (Low False Positives) |
| **Decision Tree** | Supervised Tree | 0.5957 | 0.5185 | 0.7000 | 0.8433 | Interpretable White-box Baseline |
| **Isolation Forest** | Unsupervised Anomaly | 0.0741 | 0.1429 | 0.0500 | N/A | Zero-day novel pattern detector (unlabeled) |

---

## 4. Responsible AI & Ethical Disclosure

1. **Educational Prototype:** Built for research, laboratory demonstration, and academic assessment.
2. **Transparent Provenance:** Automatically distinguishes between real Kaggle dataset evaluations and isolated synthetic demonstration runs.
3. **Presumption of Innocence:** High fraud probabilities indicate statistical anomaly risk, not legal certainty of illicit activity.
4. **Data Privacy:** Uses strictly anonymized PCA components ($V_1$ through $V_{28}$); zero PII, card numbers, or merchant identifiers are stored or exposed.
