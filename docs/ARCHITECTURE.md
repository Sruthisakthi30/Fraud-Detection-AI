# System Architecture — FraudShield AI

## 1. High-Level Architectural Overview

FraudShield AI follows a decoupled, three-tier modular architecture designed for high maintainability, enterprise security, and seamless developer onboarding:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION TIER (UI)                          │
│                                                                        │
│   HTML5 + Vanilla CSS3 (Custom Dark Financial Analytics Theme)        │
│   Vanilla JS Controller (app.js) + API Client (api.js)                 │
│   Chart.js Dynamic Visualizations (charts.js)                          │
│                                                                        │
│   10 Integrated Dashboard Views:                                      │
│   • Executive Overview        • Transaction Explorer                   │
│   • EDA Visualizer            • Fraud Prediction Inference             │
│   • Anomaly Detection Studio  • Model Laboratory                       │
│   • Explainable AI (SHAP)     • Business Intelligence                  │
│   • Investigation Cases (Ops) • System Health Monitor                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION TIER (API)                         │
│                                                                        │
│   Flask REST Service (backend/app.py & backend/routes.py)              │
│   • CORS-enabled API routes (/api/*)                                   │
│   • Strict Request Validation & Sanitization                           │
│   • Safe JSON Serialization & Error Handling                           │
│                                                                        │
│   Service & Persistence Modules:                                       │
│   • backend/services.py (Singleton Model & Artifact Caching)           │
│   • backend/cases.py (SQLite Investigation Persistence)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Python Native / Joblib / NumPy
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING & DATA ENGINE                      │
│                                                                        │
│   Data Loader (src/data_loader.py)                                     │
│   • Primary: data/creditcard.csv                                       │
│   • Demo:    data/demo_synthetic/synthetic_demo_creditcard.csv         │
│                                                                        │
│   Preprocessing & Modeling:                                            │
│   • src/preprocessing.py (Stratified Split, Scaler, Isolated SMOTE)    │
│   • src/train_models.py (LR, DT, RF, XGBoost, Isolation Forest)        │
│   • src/evaluate_models.py (F1, Precision, Recall, ROC-AUC, CM)       │
│   • src/explainability.py (SHAP Explainer & Waterfall Attributions)   │
│   • src/anomaly_detection.py (Unsupervised Isolation Forest Scorer)    │
│   • src/business_insights.py (Financial Cost/Trade-off Modeler)        │
│                                                                        │
│   Storage & Artifacts:                                                 │
│   • models/ (scaler.joblib, *.joblib, X_test.npy, y_test.npy)          │
│   • results/reports/ (model_comparison.json, eda_cache.json)           │
│   • results/cases.db (SQLite Database)                                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & Responsibilities

```
Fraud-Detection-AI/
├── backend/                  # Application Server Layer
│   ├── __init__.py           # Package marker
│   ├── app.py                # Flask entry point & static file server
│   ├── routes.py             # REST API routes & schema validators
│   ├── services.py           # Model loading, artifact caches, transaction slicer
│   └── cases.py              # SQLite case management DAO (Section H)
├── src/                      # Machine Learning Core Logic
│   ├── __init__.py           # Package marker
│   ├── config.py             # Central configuration constants & hyperparams
│   ├── data_loader.py        # Dataset validation, Kaggle loader, demo generator
│   ├── preprocessing.py      # Stratified splitter, scaler fitting, isolated SMOTE
│   ├── eda.py                # Statistical distributions & correlation heatmaps
│   ├── train_models.py       # Classifier model builders & training loops
│   ├── evaluate_models.py    # Metric calculation, CM matrices, threshold analysis
│   ├── anomaly_detection.py  # Isolation Forest unsupervised scorer
│   ├── explainability.py     # SHAP tree/linear explainers & waterfall decomposition
│   ├── business_insights.py  # Cost matrix modeler, ROI, & operational trade-offs
│   └── utils.py              # Logging setup, UTC timestamps, JSON serializers
├── frontend/                 # Client UI Tier
│   ├── index.html            # Single-Page Application (SPA) dashboard layout
│   ├── css/
│   │   └── style.css         # Dark theme financial CSS design system
│   └── js/
│       ├── api.js            # REST client wrapper for Flask backend
│       ├── charts.js         # Chart.js renderers with memory leak prevention
│       └── app.js            # SPA page navigation controller & event handlers
├── data/                     # Data Partitioning
│   ├── .gitkeep              # Git anchor
│   ├── creditcard.csv        # Primary Kaggle dataset (user placed, not in git)
│   └── demo_synthetic/       # Isolated synthetic demonstration dataset
├── models/                   # Persisted Machine Learning Artifacts
│   ├── scaler.joblib         # Fitted StandardScaler
│   ├── feature_names.json    # JSON list of feature names
│   ├── *.joblib              # Persisted trained model weights
│   ├── X_test.npy            # Persisted test feature matrix
│   └── y_test.npy            # Persisted test ground-truth labels
├── results/                  # Evaluation Outputs & Case Records
│   ├── reports/              # model_comparison.json, eda_cache.json
│   ├── figures/              # Generated diagnostic plots
│   └── cases.db              # SQLite database for investigation cases
├── docs/                     # Academic Documentation
│   ├── LITERATURE_SURVEY.md
│   ├── METHODOLOGY.md
│   ├── ARCHITECTURE.md
│   ├── RESULTS_ANALYSIS.md
│   ├── DEMO_GUIDE.md
│   └── PHASE_TRACEABILITY.md
├── tests/                    # Automated Test Suite (Pytest)
│   ├── conftest.py           # Shared session fixtures & synthetic fixtures
│   ├── test_data_loader.py   # Dataset validation & missing CSV tests
│   ├── test_preprocessing.py # SMOTE isolation & split consistency tests
│   ├── test_models.py        # Model training, metric ranges, persistence tests
│   └── test_api.py           # API endpoints, prediction validation, cases CRUD
├── train_pipeline.py         # End-to-end training pipeline CLI
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusion specifications
├── run.bat                   # Automated Windows launcher
├── PROJECT_SUMMARY.md        # Executive summary
├── VIVA_NOTES.md             # Postgraduate viva examination preparation
├── FINAL_REPORT.md           # Structured technical report template
└── README.md                 # Primary project documentation
```

---

## 3. Backend REST API Contract

| Endpoint | Method | Description | Response Status |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Health check, timestamp, and dataset status | `200 OK` |
| `/api/dataset/summary` | `GET` | Dataset metadata, row counts, fraud percentages | `200 OK`, `404 Not Found` |
| `/api/eda` | `GET` | Cached statistical histograms, time & amount distributions | `200 OK`, `404 Not Found` |
| `/api/eda/feature` | `GET` | Single-feature distribution separated by class | `200 OK`, `400 Bad Request` |
| `/api/models/status` | `GET` | Trained model artifact availability and best model meta | `200 OK` |
| `/api/models/comparison` | `GET` | Full model evaluation table, metrics, and rankings | `200 OK`, `404 Not Found` |
| `/api/predict` | `POST` | Real-time transaction inference with risk scoring | `200 OK`, `400 Bad Request`, `503 Unavailable` |
| `/api/explain` | `POST` | Local SHAP waterfall feature attributions for a transaction | `200 OK`, `400 Bad Request`, `503 Unavailable` |
| `/api/explain/global` | `GET` | Global top-10 SHAP feature importance rankings | `200 OK`, `503 Unavailable` |
| `/api/anomaly` | `GET` | Isolation Forest top-50 anomalous transactions | `200 OK`, `404 Not Found` |
| `/api/business-insights` | `GET` | Financial risk analysis, cost savings, and trade-offs | `200 OK`, `404 Not Found` |
| `/api/transactions` | `GET` | Paginated transaction explorer with search and filter | `200 OK` |
| `/api/threshold-analysis`| `GET` | Decision threshold sweeps (0.10 to 0.90) | `200 OK`, `503 Unavailable` |
| `/api/cases` | `GET` | Fetch investigation cases with status & risk filters | `200 OK` |
| `/api/cases` | `POST` | Create a new case from flagged inference or manual audit | `201 Created`, `400 Bad Request` |
| `/api/cases/<id>` | `PATCH` | Update case status, notes, or assigned severity | `200 OK`, `404 Not Found` |
| `/api/cases/<id>` | `DELETE`| Delete an investigation case | `200 OK`, `404 Not Found` |

---

## 4. Frontend Design Architecture

- **Theme & Aesthetics:** Deep navy/charcoal backdrop (`#0a0f1d`, `#111827`), high-contrast slate typography (`#e2e8f0`), electric blue accent (`#3b82f6`), vibrant semantic colors (Emerald `#10b981` for legitimate, Rose `#ef4444` for fraud, Amber `#f59e0b` for warnings).
- **Component Lifecycle:** Dynamic page lazy-loading: views only query backend endpoints when navigated to, avoiding unnecessary overhead.
- **Chart Management:** Chart wrappers automatically invoke `_destroyChart(id)` before instantiation, preventing canvas reuse memory leaks and chart jitter.
- **Responsive Layout:** CSS Grid and Flexbox with a collapsible mobile drawer navigation for phone and tablet form factors.

---

## 5. Security & Responsible AI Standards

1. **Input Validation:** Strict type casting and key validation prevents SQL/NoSQL injections and NaN/Infinity numeric crashes.
2. **Zero PII Exposure:** The dataset and inference APIs operate solely on numeric PCA vectors without card numbers or user identities.
3. **Transparent Uncertainty Disclaimers:** All inference outputs display explicit disclaimers: *"Experimental model output. Not a legally confirmed fraud finding."*
4. **Local Persistence Safety:** SQLite investigation database writes use parameterized queries (`?`), preventing SQL injection.
