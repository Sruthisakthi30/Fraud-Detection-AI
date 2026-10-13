# Phase Traceability Matrix — FraudShield AI

This matrix establishes 1-to-1 traceability between all CAT assessment criteria (Phase I, Phase II, Phase III) and the underlying implementation files, runtime artifacts, and demonstration procedures.

---

## Phase I: Conceptualization & Research Grounding

| Assessment Component | Implemented Feature | Primary Source Files | Verifiable Output / Artifact | Demonstration Procedure |
| :--- | :--- | :--- | :--- | :--- |
| **Problem Identification** | High-consequence financial chargeback & revenue leakage identification | `README.md`, `docs/METHODOLOGY.md` | Problem statement and cost matrix equations ($C_{\text{FN}}, C_{\text{FP}}$) | Present Executive Overview and Business Intelligence cost breakdown |
| **Problem Statement** | Mathematical formulation of imbalanced binary classification ($< 0.2\%$ positive class) | `docs/METHODOLOGY.md`, `src/config.py` | Formulation of F1, PR-AUC, and cost equations | Review Methodology document Section 1 |
| **Objectives** | 5 core engineering & analytical project goals | `README.md`, `PROJECT_SUMMARY.md` | Documented project objectives | Review README.md Objectives section |
| **Literature Survey** | Review of 5 foundational papers (Chawla, Dal Pozzolo, Chen, Liu, Lundberg) with verified DOIs | `docs/LITERATURE_SURVEY.md` | Comparative research table and bibliography | Review Literature Survey document Section 2 & 3 |
| **Research Gap** | Identification of 4 critical flaws in prior implementations (data leakage, lack of XAI) | `docs/LITERATURE_SURVEY.md` | Documented gap analysis | Present Literature Survey Section 4 |
| **Proposed Methodology** | Modular 3-tier architecture with leakage-free SMOTE and dual ML paradigms | `docs/METHODOLOGY.md`, `docs/ARCHITECTURE.md` | Pipeline flowchart and architecture diagram | Present Architecture document Section 1 |
| **References** | Complete academic citations with active DOIs and venue specifications | `docs/LITERATURE_SURVEY.md`, `README.md` | Peer-reviewed citation registry | Inspect References in Literature Survey |

---

## Phase II: Implementation, ML Engineering & Analytics

| Assessment Component | Implemented Feature | Primary Source Files | Verifiable Output / Artifact | Demonstration Procedure |
| :--- | :--- | :--- | :--- | :--- |
| **UI & Backend Connectivity** | REST API communication between Flask and single-page dashboard | `backend/app.py`, `backend/routes.py`, `frontend/js/api.js` | 17 active API endpoints; health dot indicator in sidebar | Launch backend, verify green health dot in UI, test API calls |
| **Data Preprocessing** | Stratified split, StandardScaler fitted strictly on train set, isolated SMOTE | `src/preprocessing.py`, `tests/test_preprocessing.py` | `models/scaler.joblib`, `models/feature_names.json` | Run `pytest tests/test_preprocessing.py` (9 tests passing) |
| **Exploratory Data Analysis** | Class imbalance charts, transaction amount histograms, correlation heatmaps | `src/eda.py`, `backend/routes.py`, `frontend/js/charts.js` | `results/reports/eda_cache.json`, interactive charts in UI | Open "Data Analysis (EDA)" view in dashboard |
| **Model Building** | 4 Supervised Classifiers (LR, DT, RF, XGBoost) + 1 Unsupervised (Isolation Forest) | `src/train_models.py`, `train_pipeline.py` | `models/*.joblib`, `models/X_test.npy`, `models/y_test.npy` | Run `python train_pipeline.py --demo`; observe training progression |
| **Model Evaluation** | F1, Precision, Recall, ROC-AUC, PR-AUC, Confusion Matrix calculation | `src/evaluate_models.py`, `tests/test_models.py` | `results/reports/model_comparison.json` | Open "Model Laboratory" view in dashboard; inspect comparison table |
| **Explainable AI (XAI)** | Global feature importance rankings & real-time local transaction SHAP waterfall | `src/explainability.py`, `frontend/js/app.js` | Interactive SHAP bar waterfall rendering | Run prediction, observe green/red SHAP feature impact bars |
| **Anomaly Detection** | Unsupervised Isolation Forest score ranking & thresholding | `src/anomaly_detection.py`, `frontend/js/app.js` | Top-50 ranked anomaly table with anomaly scores | Open "Anomaly Detection" view in dashboard |
| **Business Insights** | Cost matrix modeling, financial chargeback savings, threshold analysis | `src/business_insights.py`, `docs/RESULTS_ANALYSIS.md` | 89.0% cost reduction analysis; threshold sweep curves | Open "Business Intelligence" view in dashboard |
| **Investigation Operations** | SQLite-persisted fraud case management workspace (Section H) | `backend/cases.py`, `backend/routes.py`, `frontend/js/app.js` | `results/cases.db`, Case CRUD workflows | Create, edit, and escalate investigation cases in UI |

---

## Phase III: Deployment, Testing & Viva Readiness

| Assessment Component | Implemented Feature | Primary Source Files | Verifiable Output / Artifact | Demonstration Procedure |
| :--- | :--- | :--- | :--- | :--- |
| **Deployment & Local Startup** | Reproducible Windows startup batch script & PowerShell instructions | `run.bat`, `README.md`, `.env.example` | Automated single-click execution | Double click `run.bat` or run commands in PowerShell |
| **Automated Testing** | Comprehensive 55-test suite across preprocessing, loader, models, and API | `tests/conftest.py`, `tests/*.py` | `pytest tests/` (100% pass rate in < 6 s) | Execute `python -m pytest tests/` in terminal |
| **Presentation & Viva Prep** | Structured examination Q&A guide grounded in implementation code | `VIVA_NOTES.md`, `docs/DEMO_GUIDE.md` | 15+ comprehensive technical viva questions and answers | Review VIVA_NOTES.md with examiners |
| **GitHub Readiness** | Clean git structure, `.gitignore`, dependency specs, zero secrets committed | `.gitignore`, `requirements.txt`, `README.md` | Repository ready for clean git push | Inspect `.gitignore` and run status check |
| **Final Documentation** | Full-scale technical report and executive project overview | `FINAL_REPORT.md`, `PROJECT_SUMMARY.md` | Complete academic report artifacts | Review Final Report document |
