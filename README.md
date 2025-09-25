🕵️‍♂️ FraudRadar: End-to-End Fraud Detection Engine
FraudRadar is an end-to-end project notebook for building and explaining a high-performance fraud detection model.
It demonstrates a complete, MLOps-ready workflow on a large-scale dataset, covering everything from feature engineering and time-aware validation to deep model explainability.

📌 Features
✅ End-to-End Notebook – All steps from data loading to recommendations in a single, reproducible workflow.

✅ Advanced Feature Engineering – Creates powerful predictors like balance deltas, transaction mismatches, and zero-balance anomalies.

✅ Time-Aware Validation – Uses a strict time-based data split to prevent data leakage and simulate real-world model performance.

✅ High-Performance Modeling – Leverages LightGBM and XGBoost, optimized for speed and accuracy on tabular data.

✅ Deep Model Explainability – Integrates SHAP to provide both global (overall drivers) and local (per-transaction) explanations.

✅ Business-Centric Optimization – Includes logic to tune the decision threshold based on business goals like achieving a target precision rate.

🔄 Project Workflow
[Data Ingestion] -> [EDA & Cleaning] -> [Feature Engineering] -> [Time-Aware Split] -> [Model Training (LGBM)] -> [Evaluation] -> [Explainability (SHAP)] -> [Recommendations]
⚙️ Tech Stack
Core Libraries: Python, Pandas, NumPy

Modeling: Scikit-learn, LightGBM, XGBoost

Explainability: SHAP

Visualization: Matplotlib

Statistics: Statsmodels (for VIF)

Development: Jupyter Notebook

🚀 Getting Started
1️⃣ Prerequisites
Python 3.9+

A virtual environment manager like venv or conda.

2️⃣ Setup
Clone the Repository

Bash

git clone https://github.com/bhardwaj-yash1/Fraud-detection-model.git
cd Fraud-detection-model
📂 Create a Virtual Environment

Bash

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
📦 Install Dependencies

Bash

pip install -r requirements.txt
(Note: A requirements.txt file should be created containing the packages from the notebook's install cell).

💾 Dataset

Download the dataset from Kaggle: Synthetic Financial Datasets for Fraud Detection.

Place the Fraud.csv file inside a ./data/ directory at the project root.

✏️ Configure Path

Open Fraud_Detection_Assignment.ipynb.

In the CONFIG cell, update the DATA_PATH variable to point to your dataset:

Python

DATA_PATH = "./data/Fraud.csv"
3️⃣ Launch
Start the Jupyter Server

Bash

jupyter notebook
Open and Run the Notebook

Open Fraud_Detection_Assignment.ipynb in your browser.

Run the cells sequentially from top to bottom to execute the entire pipeline.

📊 Example Insight from SHAP
The explainability module provides clear insights into why a transaction is flagged as fraudulent.

🤖 Fraud Prediction Explained:

For a high-risk transaction, the model's decision is driven by:

is_type_TRANSFER = 1: The transaction is a 'TRANSFER'.

oldbalanceOrg is high: The originating account was nearly drained.

mismatch_orig = 1: The amount transferred doesn't match the account's balance change.

This combination strongly aligns with the known fraud pattern of an account takeover and liquidation.

🛠️ Key Methodologies
Memory Optimization: Explicit dtype definitions are used when loading data to reduce RAM usage by over 50%.

Imbalance Handling: The scale_pos_weight parameter in LightGBM is used to give more importance to the minority (fraud) class during training.

Leakage Prevention: The time-aware split is critical. A random split would leak future information into the training set, leading to an over-optimistic and unrealistic performance evaluation.

📌 Roadmap
[ ] Hyperparameter Tuning: Integrate Optuna or Hyperopt for automated model tuning.

[ ] API Deployment: Containerize the model and prediction logic into a FastAPI service.

[ ] Interactive UI: Develop a simple Streamlit or Gradio dashboard for real-time, single-transaction predictions.

[ ] Graph Features: Experiment with graph-based features (e.g., using PyG) to detect networks of fraudulent accounts.
