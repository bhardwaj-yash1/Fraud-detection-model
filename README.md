# 🛡️ FraudGuard: End-to-End Fraud Detection Engine

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Modeling-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-9cf)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **FraudGuard** is a comprehensive, end-to-end machine learning project designed to **detect fraudulent financial transactions**. It leverages advanced feature engineering, high-performance gradient boosting models, and in-depth model explainability to identify and analyze key drivers of fraud.

---

## 📌 Features

- ✅ **Complete Data Science Pipeline** – From data cleaning and EDA to model deployment and evaluation.
- ✅ **Advanced Feature Engineering** – Creates insightful predictors like balance deltas and transaction type indicators.
- ✅ **High-Performance Models** – Implements and compares **XGBoost** and **LightGBM** for superior accuracy.
- ✅ **Time-Aware Validation** – Uses a realistic time-based split to prevent data leakage and evaluate performance on future data.
- ✅ **Deep Model Explainability** – Integrates **SHAP** and feature importance plots to understand *why* a transaction is flagged as fraudulent.
- ✅ **Business-Focused Analysis** – Delivers actionable insights, prevention recommendations, and success metrics.

---

## 🏗️ Project Workflow

Data Loading & Cleaning → Feature Engineering → Time-Aware Split → Model Training (XGBoost & LightGBM) → Evaluation → Explainability & Insights → Business Recommendations

---

## ⚙️ Tech Stack

-   **Core Libraries**: `Python`, `Pandas`, `NumPy`
-   **Machine Learning**: `Scikit-learn`, `XGBoost`, `LightGBM`
-   **Explainability**: `SHAP`
-   **Data Visualization**: `Matplotlib`, `Plotly`
-   **Environment**: `Jupyter Notebook`

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

-   Python 3.8+
-   `pip` package manager

### 2️⃣ Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Place the dataset:**
    -   Download the dataset (`Fraud.csv`).
    -   Place it inside a `dataset` folder in the project's root directory: `./dataset/Fraud.csv`.

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib plotly fastparquet
    ```

### 3️⃣ Running the Analysis

The entire project is contained within a single Jupyter Notebook.

1.  Launch Jupyter Notebook or open the file in an IDE like VS Code.
2.  Open the notebook: `Fraud Detection Assignment — End‑to‑End Solution.ipynb`.
3.  Run the cells sequentially from top to bottom to reproduce the entire workflow.

---

## 📊 Performance Highlights

The **XGBoost** model emerged as the top performer, achieving near-perfect results on the time-aware validation set.

-   **AUC**: **0.99999**
-   **Average Precision (PR AUC)**: **0.99665**
-   **F1-Score (Fraud Class)**: **0.9683**

---

## 💡 Key Findings & Recommendations

### Key Fraud Drivers

The model's explainability analysis (SHAP) revealed that the most significant predictors of fraud are:
1.  **Balance Mismatches (`deltaOrig`, `mismatch_orig`)**: Discrepancies between the transaction amount and the change in the sender's account balance.
2.  **Transaction Type**: `TRANSFER` and `CASH_OUT` are overwhelmingly the types used for fraudulent activities.
3.  **Old Balance of Origin (`oldbalanceOrg`)**: Accounts being completely drained are a strong signal of fraud.
4.  **Transaction Amount**: Unusually large transaction amounts.

### Prevention Recommendations

Based on the model's findings, the following prevention strategies are recommended:
-   **Real-time Anomaly Alerts**: Trigger alerts for transactions with significant balance mismatches.
-   **Rate Limiting**: Implement velocity rules to limit the frequency and amount of `TRANSFER` and `CASH_OUT` transactions from a single account in a short period.
-   **Multi-Factor Authentication (MFA)**: Require additional verification for high-value transfers, especially if the sender's balance is close to zero post-transaction.
-   **Graph-Based Analysis**: Use network analysis to identify and monitor potential "mule accounts" that receive funds from multiple fraudulent sources.
