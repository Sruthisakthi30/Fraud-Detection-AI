# Literature Survey — Intelligent Financial Fraud Detection

## 1. Introduction & Context

Financial fraud represents an escalating systemic risk in global digital payments, with annual worldwide losses exceeding tens of billions of dollars. Detecting fraudulent transactions presents unique machine learning challenges:
1. **Extreme Class Imbalance:** Fraudulent transactions typically constitute less than 0.2% of total transaction volume.
2. **Concept Drift & Adversarial Adaptation:** Fraudsters continuously evolve tactics to circumvent heuristic rules.
3. **Black-box Model Opacity:** High-performing models (such as deep neural networks or gradient-boosted trees) lack intrinsic interpretability, complicating compliance with global regulatory mandates (e.g., GDPR Article 22 "Right to Explanation" and the EU AI Act).

This survey evaluates key foundational research spanning class imbalance remediation, ensemble learning, unsupervised anomaly detection, and explainable AI (XAI).

---

## 2. Comprehensive Survey of Key Literature

### Paper 1: Handling Class Imbalance with SMOTE
- **Authors:** Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer
- **Title:** *SMOTE: Synthetic Minority Over-sampling Technique*
- **Venue:** Journal of Artificial Intelligence Research (JAIR), Vol. 16, pp. 321–357 (2002)
- **DOI:** [10.1613/jair.953](https://doi.org/10.1613/jair.953)
- **Key Contribution:** Introduced SMOTE, an algorithm that generates synthetic minority class examples along feature space line segments joining $k$-nearest neighbors, rather than simple replication.
- **Critical Analysis & Relevance:** SMOTE avoids the overfitting inherent in simple random oversampling. However, applying SMOTE indiscriminately across the whole dataset causes severe data leakage. In FraudShield AI, SMOTE is strictly restricted to the training partition after stratified splitting.

### Paper 2: European Credit Card Benchmark & Adaptive Fraud Detection
- **Authors:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi
- **Title:** *Calibrating Probability with Undersampling for Unbalanced Classification*
- **Venue:** IEEE Symposium Series on Computational Intelligence (SSCI), Cape Town, South Africa, pp. 159–166 (2015)
- **DOI:** [10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33)
- **Key Contribution:** Released and benchmarked the European Credit Card Fraud Dataset (284,807 transactions with 492 frauds, 0.172% positive class), proposing probability calibration techniques to correct for artificial undersampling/oversampling bias.
- **Critical Analysis & Relevance:** Serves as the primary empirical benchmark for FraudShield AI. Proves that accuracy is an invalid metric in fraud detection, highlighting the necessity of Precision-Recall curves, F1-scores, and cost-sensitive threshold analysis.

### Paper 3: Extreme Gradient Boosting (XGBoost)
- **Authors:** Tianqi Chen, Carlos Guestrin
- **Title:** *XGBoost: A Scalable Tree Boosting System*
- **Venue:** Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), pp. 785–794 (2016)
- **DOI:** [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- **Key Contribution:** Developed a regularized, cache-aware gradient boosting framework incorporating second-order Taylor approximations, column subsampling, and shrinkage.
- **Critical Analysis & Relevance:** Demonstrates state-of-the-art non-linear classification performance on tabular tabular financial data, effectively capturing high-order feature interactions among PCA latent components $V_1$ through $V_{28}$.

### Paper 4: Unsupervised Anomaly Detection via Isolation Forest
- **Authors:** Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou
- **Title:** *Isolation Forest*
- **Venue:** 8th IEEE International Conference on Data Mining (ICDM '08), Pisa, Italy, pp. 413–422 (2008)
- **DOI:** [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
- **Key Contribution:** Replaced distance and density-based anomaly estimation with isolation trees ($iTrees$). Anomalies are isolated near tree roots with significantly shorter average path lengths $h(x)$.
- **Critical Analysis & Relevance:** Allows detection of novel zero-day fraud mechanisms where ground-truth labels do not exist. FraudShield AI integrates Isolation Forest alongside supervised classifiers to compare unsupervised anomaly flagging against supervised discriminative learning.

### Paper 5: A Unified Approach to Interpreting Model Predictions (SHAP)
- **Authors:** Scott M. Lundberg, Su-In Lee
- **Title:** *A Unified Approach to Interpreting Model Predictions*
- **Venue:** Advances in Neural Information Processing Systems (NeurIPS 30), pp. 4765–4774 (2017)
- **URL:** [Curran Associates / NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **Key Contribution:** Unified LIME, DeepLIFT, and Shapley values through SHAP (SHapley Additive exPlanations), proving that Shapley values provide the unique solution satisfying Local Accuracy, Missingness, and Consistency.
- **Critical Analysis & Relevance:** Solves the explainability requirement for high-stakes fraud scoring. FraudShield AI uses TreeSHAP and Kernel/Linear SHAP to generate both global feature attribution rankings and real-time local transaction waterfall contributions.

---

## 3. Comparative Literature Summary

| Study | Primary Methodology | Imbalance Strategy | Explainability | Key Advantage | Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chawla et al. (2002)** | Synthetic Oversampling | $k$-NN synthetic generation | None | Balances decision boundaries | Risk of noise synthesis in overlapping regions |
| **Dal Pozzolo et al. (2015)** | Ensemble Classifiers | Calibrated Resampling | None | Realistic latency & fraud prevalence | Limited to specific distribution assumptions |
| **Chen & Guestrin (2016)** | Gradient Tree Boosting | `scale_pos_weight` & regularization | Feature importance (gain/cover) | Exceptional tabular performance | Global gain can be biased towards high-cardinality features |
| **Liu et al. (2008)** | Tree Isolation Anomaly Detection | Unsupervised | Path length score | Linear time complexity, no labels needed | Higher false positive rate on dense legitimate regions |
| **Lundberg & Lee (2017)** | Cooperative Game Theory (Shapley) | Model-agnostic & tree-specific | Game-theoretic feature attributions | Theoretically grounded local & global XAI | KernelSHAP background sampling latency on large sets |

---

## 4. Research Gap

Prior implementations and existing literature frequently exhibit significant practical limitations:
1. **Data Leakage in Academic Prototypes:** Numerous published implementations apply normalization and SMOTE before the train-test split, artificially inflating reported test F1 and AUC scores.
2. **Siloed Supervised vs. Unsupervised Systems:** Systems typically deploy *either* supervised classifiers (failing on novel attack patterns) *or* unsupervised anomaly detectors (suffering excessive false positives). There is a lack of integrated dashboards comparing both paradigms simultaneously on identical test splits.
3. **Absence of Actionable Explainability:** Model explainability is frequently treated as an offline static figure rather than an interactive, per-transaction forensic capability linked to an audit workspace.
4. **Lack of Operational Triage Integration:** Few educational or laboratory projects provide an end-to-end bridge between ML risk probabilities and an investigator case management workspace with state persistence.

---

## 5. Proposed Contribution of FraudShield AI

FraudShield AI directly addresses these research gaps by delivering:
- **Strict Leakage-Free Pipeline:** Preprocessing (scaling and SMOTE) fitted exclusively on stratified training folds, ensuring test partition integrity.
- **Unified Dual-Paradigm Laboratory:** Side-by-side evaluation of 4 supervised classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) and 1 unsupervised detector (Isolation Forest).
- **Interactive Game-Theoretic XAI:** Live, per-transaction SHAP feature attributions explaining model decisions with directional positive/negative impact analysis.
- **Operational Case Management:** SQLite-persisted fraud investigation registry translating statistical predictions into accountable auditor workflows.
- **Full Traceability & Provenance:** Honest reporting distinguishing between real Kaggle benchmark evaluations and synthetic demonstration runs.
