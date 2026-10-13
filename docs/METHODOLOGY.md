# Methodology — FraudShield AI

## 1. Problem Statement

Digital financial payment networks process millions of transactions per day, with fraudulent attempts accounting for a minute fraction ($< 0.2\%$). Failure to detect fraudulent transactions results in catastrophic financial chargebacks, merchant liability, and reputational damage. Conversely, incorrectly declining legitimate transactions ("False Positives") degrades customer trust and causes revenue leakage.

The core engineering and data science challenge is to construct an automated detection engine that:
1. Maximizes minority fraud detection (**Recall** $\ge 0.85$),
2. Minimizes disruption to legitimate cardholders (**Precision** $\ge 0.80$),
3. Operates within low latency constraints ($< 50\,\text{ms}$ per prediction),
4. Delivers transparent explanations for every flagged decision (**Explainable AI** via SHAP).

---

## 2. Dataset Specification

### 2.1 Schema & Latent Features
FraudShield AI is designed around the European Credit Card Fraud Detection benchmark (September 2013 transactions by European cardholders):
- **Observations:** 284,807 transactions.
- **Target Variable:** `Class` $\in \{0, 1\}$ (0 = Legitimate: 284,315 [99.828%], 1 = Fraudulent: 492 [0.172%]).
- **Temporal Feature:** `Time` — elapsed seconds since the first recorded transaction in the dataset.
- **Monetary Feature:** `Amount` — transaction value in Euros ($€$).
- **Latent Features:** $V_1, V_2, \dots, V_{28}$ — orthogonal principal components extracted via Principal Component Analysis (PCA) to protect cardholder privacy and confidentiality.
- **Privacy Assurance:** No personally identifiable information (PII), card numbers, CVVs, or merchant identities are stored or required.

---

## 3. Preprocessing & Leakage Prevention Pipeline

Data leakage is the most prevalent flaw in financial machine learning experiments. FraudShield AI enforces an airtight pipeline architecture:

```
Raw Data (N Rows)
      │
      ▼
┌──────────────────────────────────────────────┐
│  Stratified Split (80% Train, 20% Test)       │
│  (Random Seed = 42, Preserves Class Ratio)   │
└──────────────────────────────────────────────┘
      │                                │
      ▼                                ▼
[Train Partition: 80%]            [Test Partition: 20%]
      │                                │
      ▼                                │
Fit StandardScaler                     │
Transform Train Features               │
      │                                │
      ▼                                │
Apply SMOTE (Train Only)               ▼
Resample Minority Class          Transform Test Features
      │                          (Using Pre-fitted Scaler)
      ▼                                │
X_train_res, y_train_res               ▼
      │                          X_test, y_test
      │                         (UNTOUCHED by SMOTE)
      ▼                                │
Train Models                           ▼
(LR, DT, RF, XGBoost)            Evaluate Models
```

### 3.1 Stratified Train/Test Split
A stratified 80/20 train/test split partitions the data, guaranteeing that both sets maintain identical fraud ratios.

### 3.2 Standard Scaling
Standardization scales features to zero mean and unit variance:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
$\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed strictly on the training partition and applied identically to the test partition and live inference requests.

### 3.3 SMOTE Application
Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training set. Synthetic fraud vectors are generated via linear interpolation between neighboring fraud instances:
$$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$
The test partition remains completely pristine and representative of real-world imbalanced production distributions.

---

## 4. Exploratory Data Analysis (EDA)

The integrated EDA module computes statistical distributions across:
1. **Class Distribution:** Quantifies extreme class skew ($99.83\%$ vs $0.17\%$).
2. **Transaction Amount Analysis:** Fraud amounts cluster around low-to-medium denominations with specific outliers, whereas legitimate amounts follow a heavy-tailed distribution.
3. **Temporal Dynamics:** Cyclic transaction volume patterns over a 48-hour window, contrasting daytime legit volume peaks with consistent fraud attempts during low-traffic off-hours.
4. **Feature Correlations:** Correlation analysis identifying key latent components with significant positive ($V_4, V_11$) and negative ($V_{12}, V_{14}, V_{17}$) correlation with fraud occurrences.

---

## 5. Machine Learning Models

FraudShield AI implements five distinct models spanning supervised classification and unsupervised anomaly detection:

### 5.1 Supervised Classifiers
1. **Logistic Regression:**
   - Linear baseline utilizing the sigmoid activation function:
     $$P(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w}^T x + b)}}$$
   - Tuned with `class_weight='balanced'`, L2 regularization (`C=1.0`), and L-BFGS solver.
2. **Decision Tree Classifier:**
   - Non-linear rule-based tree model using Gini impurity splitting:
     $$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$
   - Constrained to `max_depth=10` to avoid over-specialization on synthetic minority points.
3. **Random Forest Classifier:**
   - Bootstrap aggregating ensemble of 100 decorrelated decision trees with random feature sub-selection.
   - Robust against noise and high-variance outliers; `max_depth=12`, parallelized execution (`n_jobs=-1`).
4. **XGBoost (Extreme Gradient Boosted Trees):**
   - Sequentially boosted decision trees optimizing second-order Taylor expansion of the objective function:
     $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
   - Configured with `n_estimators=150`, `learning_rate=0.1`, `max_depth=6`, and `subsample=0.8`.

### 5.2 Unsupervised Anomaly Detection
5. **Isolation Forest:**
   - Construct an ensemble of Isolation Trees ($iTrees$) over unscaled raw features.
   - Calculates path lengths $h(x)$ to isolate points; anomalous instances isolate close to root nodes:
     $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
   - Serves as an unsupervised zero-day anomaly detector independent of supervised labels.

---

## 6. Evaluation Framework & Metrics

In extreme class imbalance, standard **Accuracy** is dangerously misleading (a naive model predicting all transactions as legitimate achieves $99.83\%$ accuracy while detecting 0 frauds). FraudShield AI prioritizes:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(t) \, d(\text{FPR}(t))$$

- **Primary Selection Criterion:** The best supervised model is chosen based on **F1-Score** on the un-resampled test partition, balancing customer friction against financial losses.

---

## 7. Explainable AI Methodology (SHAP)

To satisfy algorithmic transparency requirements, FraudShield AI computes Shapley Additive Explanations:
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$
- **Global Feature Importance:** Evaluates mean absolute SHAP values across test instances:
  $$\text{Importance}_j = \frac{1}{M}\sum_{k=1}^M |\phi_j^{(k)}|$$
- **Local Waterfall Attributions:** Decomposes an individual prediction into positive pushing forces (increasing fraud risk) and negative mitigating forces (decreasing fraud risk) relative to base expected value $E[f(x)]$.
- **Model Compatibility:** TreeSHAP for tree-based estimators (XGBoost, Random Forest, Decision Tree) and Linear/Kernel SHAP for linear baselines.

---

## 8. Limitations & Boundary Conditions

1. **Synthetic Data Boundaries:** While the primary dataset stems from real European transactions, PCA projection obscures raw domain semantics (e.g., MCC codes, IP addresses, geographic velocity).
2. **Dynamic Concept Drift:** Production payment fraud undergoes adversarial distribution shifts; static batch models require periodic retraining triggers.
3. **Interpretability Latency:** Computing exhaustive background SHAP explanations across millions of instances is computationally intensive, requiring background downsampling for real-time responsiveness.
