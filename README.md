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

```mermaid
graph LR
    A[Data Loading & Cleaning] --> B[Feature Engineering];
    B --> C[Time-Aware Data Split];
    C --> D["Model Training & Tuning <br>(XGBoost & LightGBM)"];
    D --> E["Performance Evaluation <br>(AUC, PR-AUC, Confusion Matrix)"];
    E --> F["Explainability & Insights <br>(SHAP Analysis)"];
    F --> G[Business Recommendations];
