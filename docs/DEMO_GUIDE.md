# CAT Live Demonstration & Viva Guide — FraudShield AI

## 1. Quick Startup Guide (Windows PowerShell)

Open PowerShell in the project directory:

```powershell
# Step 1: Navigate to project root
cd C:\Users\Hp\Documents\Fraud-Detection-AI

# Step 2: Run automated test suite (verifies 55 unit/integration tests)
python -m pytest tests/

# Step 3: Run the training pipeline (trains all 5 models & generates artifacts)
python train_pipeline.py --demo

# Step 4: Start the Flask application server
python backend/app.py

# Step 5: Open your web browser
# Navigate to: http://localhost:5000
```

*Note: Alternatively, double-click `run.bat` in Windows Explorer to run all steps automatically.*

---

## 2. Step-by-Step Viva / CAT Demonstration Script

During your viva presentation, navigate through the dashboard in this exact sequence to demonstrate full technical competence:

### Demonstration Step 1: Executive Overview & Health Status
- **Action:** Open `http://localhost:5000`. Show the Executive Overview page.
- **Key Visuals to Highlight:**
  - Dynamic KPI cards displaying Total Transactions, Legitimate count, Fraudulent count, and Fraud Percentage.
  - Data Provenance indicator (clearly indicating whether the system is loaded with the official European dataset or isolated Synthetic Demo data).
  - Best Evaluated Model badge (XGBoost, F1: 0.9000).
  - Health dot in the sidebar indicating active Flask backend connectivity.
- **Concept to Explain:** Discuss how the dashboard serves as a centralized single source of truth, loading metadata directly from persisted artifacts rather than hardcoded UI values.

### Demonstration Step 2: Transaction Explorer & Filter Controls
- **Action:** Click **Transaction Explorer** in the left sidebar.
- **Key Visuals to Highlight:**
  - Paginated transaction data table ($50$ rows per page).
  - Filter by class: Toggle "All Transactions", "Fraud Only", and "Legitimate Only".
  - Inspect high-dimensional PCA features ($V_1$ through $V_{28}$), timestamps, and monetary amounts.
- **Concept to Explain:** Explain the anonymization structure of the European dataset where PCA protects sensitive cardholder identity while preserving variance and correlation geometry.

### Demonstration Step 3: Exploratory Data Analysis (EDA)
- **Action:** Click **Data Analysis (EDA)** in the left sidebar.
- **Key Visuals to Highlight:**
  - Class Imbalance Doughnut Chart ($98\%$ vs $2\%$).
  - Amount Distribution comparison showing distinct expenditure profiles between fraudulent and legitimate events.
  - Interactive feature distribution selector (inspect $V_4, V_14$, and $V_{17}$).
- **Concept to Explain:** Explain why class imbalance renders standard accuracy metrics useless, necessitating F1, Precision-Recall AUC, and cost matrix modeling.

### Demonstration Step 4: Real-Time Fraud Prediction Inference
- **Action:** Click **Fraud Prediction** in the left sidebar.
- **Demonstration Actions:**
  1. Click **"Load Test Fraud Example"** — form fields populate with an actual test-set fraudulent vector.
  2. Click **"Run Prediction"**.
  3. Observe immediate output: "🚨 FRAUD DETECTED", Fraud probability ($95\%+$), and Risk Category: "Critical".
  4. Point out the responsible AI disclaimer: *"Experimental model output. Not a confirmed fraud finding."*
  5. Click **"Load Legit Example"** and click **"Run Prediction"** — observe switch to "✅ LEGITIMATE" (Probability $< 5\%$, Risk Category: "Low").

### Demonstration Step 5: Explainable AI (SHAP Waterfall)
- **Action:** Scroll to the SHAP explanation section on the Prediction page, or open the **Explainable AI** tab.
- **Key Visuals to Highlight:**
  - Interactive SHAP waterfall bar chart detailing the exact top-10 positive and negative contributing features.
  - Directional breakdown: Green bars pushing risk downward, red bars pushing risk upward.
- **Concept to Explain:** Explain cooperative game theory Shapley values. Emphasize that in financial services, GDPR Article 22 mandates that customers have the right to know why an algorithmic decision declined their payment.

### Demonstration Step 6: Model Laboratory & ROC Curves
- **Action:** Click **Model Laboratory** in the left sidebar.
- **Key Visuals to Highlight:**
  - Full comparative matrix across all 5 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, Isolation Forest).
  - Metric bar chart comparing F1, Precision, and Recall.
  - Interactive Confusion Matrices for each classifier.
  - ROC curve visualization demonstrating XGBoost's area under curve ($0.9996$).
- **Concept to Explain:** Contrast supervised learning (optimizing decision boundaries against known labels) with unsupervised anomaly detection.

### Demonstration Step 7: Unsupervised Anomaly Detection (Isolation Forest)
- **Action:** Click **Anomaly Detection** in the left sidebar.
- **Key Visuals to Highlight:**
  - Ranked table of the top-50 most unusual transactions scored by tree isolation depth.
  - Anomaly score metrics and decision thresholds.
- **Concept to Explain:** Detail how Isolation Forest isolates points in feature space without labels, providing zero-day detection capability for novel fraud mechanisms.

### Demonstration Step 8: Fraud Investigation Workspace (Operations)
- **Action:** Click **Investigation Cases** in the left sidebar.
- **Demonstration Actions:**
  1. Show case KPI overview (Total, Open, Under Review, Resolved).
  2. Click **"➕ New Case"** to create an investigation case (or demonstrate clicking "🚩 Log Investigation Case" directly from a flagged prediction).
  3. Update status from "Open" to "Under Review", add an investigator note, and save.
  4. Confirm that records are stored in the local SQLite database (`results/cases.db`).
- **Concept to Explain:** Explain the bridge between data science output and operational triage.

### Demonstration Step 9: Business Intelligence & Threshold Analysis
- **Action:** Click **Business Intelligence** in the left sidebar.
- **Key Visuals to Highlight:**
  - Financial cost reduction metrics ($89.0\%$ cost savings over naive baseline).
  - Threshold trade-off curve (Precision vs Recall vs F1 across thresholds $0.10$ to $0.90$).
- **Concept to Explain:** Highlight that decision thresholds must align with merchant risk tolerance—low thresholds for high-risk jewelry/electronics, high thresholds for low-friction digital goods.

---

## 3. Common Viva Questions & Model Answers

| Question | Model Answer |
| :--- | :--- |
| **Why did you use SMOTE only on the training set?** | Applying SMOTE to the test set or before splitting creates synthetic instances derived from test vectors, causing severe data leakage. Our test set reflects pristine, un-resampled production distributions. |
| **Why is accuracy a poor metric here?** | In a dataset where 99.83% of transactions are legitimate, a model that classifies everything as legitimate achieves 99.83% accuracy while catching 0 frauds. We prioritize F1-score, Precision, and Recall. |
| **How does SHAP differ from standard feature importance?** | Standard Gini/gain feature importance is global and can be biased toward high-cardinality features. SHAP is grounded in cooperative game theory, providing locally consistent, additive feature contributions for individual transactions. |
| **Why does Isolation Forest have a low F1 score compared to XGBoost?** | Isolation Forest is unsupervised and detects statistical outliers in feature space, not fraud labels. Many legitimate transactions have unusual amounts or times. Its purpose is zero-day novel anomaly discovery, not supervised discriminative classification. |
