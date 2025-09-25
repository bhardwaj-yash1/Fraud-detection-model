├── .ipynb_checkpoints
    └── Fraud_Detection_Assignment-checkpoint.ipynb
├── Executive_Summary_Template.md
├── Fraud_Detection_Assignment.ipynb
├── README.md
└── anaconda_projects
    └── db
        └── project_filebrowser.db


/.ipynb_checkpoints/Fraud_Detection_Assignment-checkpoint.ipynb:
--------------------------------------------------------------------------------
  1 | {
  2 |  "cells": [
  3 |   {
  4 |    "cell_type": "markdown",
  5 |    "id": "c52cbab6",
  6 |    "metadata": {},
  7 |    "source": [
  8 |     "\n",
  9 |     "# Fraud Detection Assignment — End‑to‑End Solution\n",
 10 |     "**Author:** _Your Name_  \n",
 11 |     "**Deadline:** 25 Aug, 11:59 PM IST\n",
 12 |     "\n",
 13 |     "This notebook is structured to satisfy all deliverables in the brief:\n",
 14 |     "\n",
 15 |     "1) **Data cleaning**: missing values, outliers, multicollinearity  \n",
 16 |     "2) **Model**: end‑to‑end build with rationale  \n",
 17 |     "3) **Feature selection rationale**  \n",
 18 |     "4) **Performance demonstration**: rigorous, time‑aware validation, metrics, plots  \n",
 19 |     "5) **Key fraud drivers** (global + local explainability)  \n",
 20 |     "6) **Sanity of factors** (business reasoning)  \n",
 21 |     "7) **Prevention recommendations**  \n",
 22 |     "8) **Measurement plan**\n",
 23 |     "\n",
 24 |     "> 🔧 **How to use**\n",
 25 |     "> - Put your dataset path in the cell below (`DATA_PATH`).  \n",
 26 |     "> - Run cells in order.  \n",
 27 |     "> - If running on Colab, enable GPU/TPU (optional).  \n",
 28 |     "> - This notebook uses memory‑efficient reading + LightGBM/XGBoost (install as needed).\n"
 29 |    ]
 30 |   },
 31 |   {
 32 |    "cell_type": "code",
 33 |    "execution_count": 2,
 34 |    "id": "477418be",
 35 |    "metadata": {},
 36 |    "outputs": [
 37 |     {
 38 |      "name": "stdout",
 39 |      "output_type": "stream",
 40 |      "text": [
 41 |       "Collecting pandas\n",
 42 |       "  Using cached pandas-2.3.2-cp313-cp313-win_amd64.whl.metadata (19 kB)\n",
 43 |       "Requirement already satisfied: numpy in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (2.2.6)\n",
 44 |       "Collecting scikit-learn\n",
 45 |       "  Using cached scikit_learn-1.7.1-cp313-cp313-win_amd64.whl.metadata (11 kB)\n",
 46 |       "Collecting lightgbm\n",
 47 |       "  Using cached lightgbm-4.6.0-py3-none-win_amd64.whl.metadata (17 kB)\n",
 48 |       "Collecting xgboost\n",
 49 |       "  Using cached xgboost-3.0.4-py3-none-win_amd64.whl.metadata (2.1 kB)\n",
 50 |       "Collecting shap\n",
 51 |       "  Using cached shap-0.48.0-cp313-cp313-win_amd64.whl.metadata (25 kB)\n",
 52 |       "Collecting matplotlib\n",
 53 |       "  Using cached matplotlib-3.10.5-cp313-cp313-win_amd64.whl.metadata (11 kB)\n",
 54 |       "Collecting plotly\n",
 55 |       "  Using cached plotly-6.3.0-py3-none-any.whl.metadata (8.5 kB)\n",
 56 |       "Requirement already satisfied: tqdm in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (4.67.1)\n",
 57 |       "Requirement already satisfied: pyarrow in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (21.0.0)\n",
 58 |       "Collecting fastparquet\n",
 59 |       "  Using cached fastparquet-2024.11.0-cp313-cp313-win_amd64.whl.metadata (4.3 kB)\n",
 60 |       "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
 61 |       "Requirement already satisfied: pytz>=2020.1 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from pandas) (2025.2)\n",
 62 |       "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from pandas) (2025.2)\n",
 63 |       "Requirement already satisfied: scipy>=1.8.0 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from scikit-learn) (1.16.1)\n",
 64 |       "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from scikit-learn) (1.5.1)\n",
 65 |       "Requirement already satisfied: threadpoolctl>=3.1.0 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from scikit-learn) (3.6.0)\n",
 66 |       "Requirement already satisfied: packaging>20.9 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from shap) (25.0)\n",
 67 |       "Requirement already satisfied: slicer==0.0.8 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from shap) (0.0.8)\n",
 68 |       "Collecting numba>=0.54 (from shap)\n",
 69 |       "  Using cached numba-0.61.2-cp313-cp313-win_amd64.whl.metadata (2.8 kB)\n",
 70 |       "Requirement already satisfied: cloudpickle in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from shap) (3.1.1)\n",
 71 |       "Requirement already satisfied: typing-extensions in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from shap) (4.14.1)\n",
 72 |       "Collecting contourpy>=1.0.1 (from matplotlib)\n",
 73 |       "  Using cached contourpy-1.3.3-cp313-cp313-win_amd64.whl.metadata (5.5 kB)\n",
 74 |       "Requirement already satisfied: cycler>=0.10 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from matplotlib) (0.12.1)\n",
 75 |       "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from matplotlib) (4.59.1)\n",
 76 |       "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from matplotlib) (1.4.9)\n",
 77 |       "Requirement already satisfied: pillow>=8 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from matplotlib) (11.3.0)\n",
 78 |       "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from matplotlib) (3.2.3)\n",
 79 |       "Requirement already satisfied: narwhals>=1.15.1 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from plotly) (2.2.0)\n",
 80 |       "Requirement already satisfied: colorama in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from tqdm) (0.4.6)\n",
 81 |       "Requirement already satisfied: cramjam>=2.3 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from fastparquet) (2.11.0)\n",
 82 |       "Requirement already satisfied: fsspec in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from fastparquet) (2025.7.0)\n",
 83 |       "Requirement already satisfied: llvmlite<0.45,>=0.44.0dev0 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from numba>=0.54->shap) (0.44.0)\n",
 84 |       "Requirement already satisfied: six>=1.5 in c:\\users\\yashh\\fraud detection model\\fraud-env\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
 85 |       "Using cached pandas-2.3.2-cp313-cp313-win_amd64.whl (11.0 MB)\n",
 86 |       "Using cached scikit_learn-1.7.1-cp313-cp313-win_amd64.whl (8.7 MB)\n",
 87 |       "Using cached lightgbm-4.6.0-py3-none-win_amd64.whl (1.5 MB)\n",
 88 |       "Using cached xgboost-3.0.4-py3-none-win_amd64.whl (56.8 MB)\n",
 89 |       "Using cached shap-0.48.0-cp313-cp313-win_amd64.whl (545 kB)\n",
 90 |       "Using cached matplotlib-3.10.5-cp313-cp313-win_amd64.whl (8.1 MB)\n",
 91 |       "Using cached plotly-6.3.0-py3-none-any.whl (9.8 MB)\n",
 92 |       "Using cached fastparquet-2024.11.0-cp313-cp313-win_amd64.whl (673 kB)\n",
 93 |       "Using cached contourpy-1.3.3-cp313-cp313-win_amd64.whl (226 kB)\n",
 94 |       "Using cached numba-0.61.2-cp313-cp313-win_amd64.whl (2.8 MB)\n",
 95 |       "Installing collected packages: plotly, numba, contourpy, xgboost, scikit-learn, pandas, matplotlib, lightgbm, shap, fastparquet\n",
 96 |       "Successfully installed contourpy-1.3.3 fastparquet-2024.11.0 lightgbm-4.6.0 matplotlib-3.10.5 numba-0.61.2 pandas-2.3.2 plotly-6.3.0 scikit-learn-1.7.1 shap-0.48.0 xgboost-3.0.4\n",
 97 |       "Note: you may need to restart the kernel to use updated packages.\n"
 98 |      ]
 99 |     },
100 |     {
101 |      "name": "stderr",
102 |      "output_type": "stream",
103 |      "text": [
104 |       "\n",
105 |       "[notice] A new release of pip is available: 25.0.1 -> 25.2\n",
106 |       "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
107 |      ]
108 |     },
109 |     {
110 |      "name": "stdout",
111 |      "output_type": "stream",
112 |      "text": [
113 |       "Packages -> LGBM: True | XGB: True | SHAP: True\n"
114 |      ]
115 |     }
116 |    ],
117 |    "source": [
118 |     "\n",
119 |     "# If running locally and missing packages, uncomment installs:\n",
120 |     "%pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib plotly tqdm pyarrow fastparquet\n",
121 |     "\n",
122 |     "import os, gc, math, warnings\n",
123 |     "warnings.filterwarnings(\"ignore\")\n",
124 |     "\n",
125 |     "import numpy as np\n",
126 |     "import pandas as pd\n",
127 |     "from tqdm import tqdm\n",
128 |     "\n",
129 |     "from sklearn.model_selection import train_test_split\n",
130 |     "from sklearn.preprocessing import StandardScaler\n",
131 |     "from sklearn.metrics import (\n",
132 |     "    roc_auc_score, average_precision_score, precision_recall_curve,\n",
133 |     "    confusion_matrix, classification_report, roc_curve\n",
134 |     ")\n",
135 |     "\n",
136 |     "import matplotlib.pyplot as plt\n",
137 |     "\n",
138 |     "# Optional (if available):\n",
139 |     "try:\n",
140 |     "    import lightgbm as lgb\n",
141 |     "    HAS_LGBM = True\n",
142 |     "except Exception:\n",
143 |     "    HAS_LGBM = False\n",
144 |     "\n",
145 |     "try:\n",
146 |     "    import xgboost as xgb\n",
147 |     "    HAS_XGB = True\n",
148 |     "except Exception:\n",
149 |     "    HAS_XGB = False\n",
150 |     "\n",
151 |     "try:\n",
152 |     "    import shap\n",
153 |     "    HAS_SHAP = True\n",
154 |     "except Exception:\n",
155 |     "    HAS_SHAP = False\n",
156 |     "\n",
157 |     "print(\"Packages -> LGBM:\", HAS_LGBM, \"| XGB:\", HAS_XGB, \"| SHAP:\", HAS_SHAP)\n"
158 |    ]
159 |   },
160 |   {
161 |    "cell_type": "code",
162 |    "execution_count": null,
163 |    "id": "7c253202",
164 |    "metadata": {},
165 |    "outputs": [],
166 |    "source": [
167 |     "\n",
168 |     "# ====== CONFIG ======\n",
169 |     "DATA_PATH = \"C:\\Users\\yashh\\Downloads\\Fraud.csv\"  # <- <-- PUT YOUR CSV PATH HERE\n",
170 |     "RANDOM_STATE = 42\n",
171 |     "TEST_SIZE = 0.20  # time-aware split is applied below; this is a fallback\n",
172 |     "TARGET_COL = \"isFraud\"\n",
173 |     "\n",
174 |     "# Memory optimization: set dtypes for known columns\n",
175 |     "DTYPES = {\n",
176 |     "    \"step\": \"int32\",\n",
177 |     "    \"type\": \"category\",\n",
178 |     "    \"amount\": \"float32\",\n",
179 |     "    \"nameOrig\": \"category\",\n",
180 |     "    \"oldbalanceOrg\": \"float32\",\n",
181 |     "    \"newbalanceOrig\": \"float32\",\n",
182 |     "    \"nameDest\": \"category\",\n",
183 |     "    \"oldbalanceDest\": \"float32\",\n",
184 |     "    \"newbalanceDest\": \"float32\",\n",
185 |     "    \"isFraud\": \"int8\",\n",
186 |     "    \"isFlaggedFraud\": \"int8\",\n",
187 |     "}\n",
188 |     "\n",
189 |     "assert os.path.exists(DATA_PATH), f\"Dataset not found at {DATA_PATH}. Please update DATA_PATH.\"\n",
190 |     "print('Using dataset at:', DATA_PATH)\n"
191 |    ]
192 |   },
193 |   {
194 |    "cell_type": "code",
195 |    "execution_count": null,
196 |    "id": "b8397175",
197 |    "metadata": {},
198 |    "outputs": [],
199 |    "source": [
200 |     "\n",
201 |     "# ====== LOAD DATA ======\n",
202 |     "# For 6.36M rows, pandas can load with the right dtypes on a 16GB+ machine.\n",
203 |     "# If RAM is tight, consider reading with chunks for EDA; we do full load for modeling.\n",
204 |     "df = pd.read_csv(DATA_PATH, dtype=DTYPES)\n",
205 |     "print(df.shape)\n",
206 |     "df.head(3)\n"
207 |    ]
208 |   },
209 |   {
210 |    "cell_type": "code",
211 |    "execution_count": null,
212 |    "id": "f42c9e70",
213 |    "metadata": {},
214 |    "outputs": [],
215 |    "source": [
216 |     "\n",
217 |     "# ====== DATA HEALTH CHECKS ======\n",
218 |     "print(\"\\nBasic Info:\")\n",
219 |     "print(df.info())\n",
220 |     "\n",
221 |     "print(\"\\nTarget balance:\")\n",
222 |     "print(df[TARGET_COL].value_counts(normalize=True).rename('proportion'))\n",
223 |     "\n",
224 |     "print(\"\\nMissing values per column:\")\n",
225 |     "print(df.isna().sum())\n",
226 |     "\n",
227 |     "# Summary stats for numeric columns\n",
228 |     "num_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
229 |     "cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()\n",
230 |     "print(\"\\nNumeric columns:\", num_cols)\n",
231 |     "print(\"Categorical columns:\", cat_cols)\n",
232 |     "\n",
233 |     "df[num_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99])\n"
234 |    ]
235 |   },
236 |   {
237 |    "cell_type": "markdown",
238 |    "id": "dd4e8482",
239 |    "metadata": {},
240 |    "source": [
241 |     "## Exploratory Data Analysis (target leakage, distributions, imbalances)"
242 |    ]
243 |   },
244 |   {
245 |    "cell_type": "code",
246 |    "execution_count": null,
247 |    "id": "618a8e98",
248 |    "metadata": {},
249 |    "outputs": [],
250 |    "source": [
251 |     "\n",
252 |     "# Target vs type\n",
253 |     "ct = pd.crosstab(df['type'], df[TARGET_COL], normalize='index')\n",
254 |     "print(ct)\n",
255 |     "\n",
256 |     "# Distribution of amounts (trimmed)\n",
257 |     "amt_q99 = df['amount'].quantile(0.99)\n",
258 |     "df['amount_clip'] = df['amount'].clip(upper=amt_q99)\n",
259 |     "df['amount_clip'].hist(bins=50)\n",
260 |     "plt.title('Amount (clipped at 99th pct)')\n",
261 |     "plt.xlabel('amount')\n",
262 |     "plt.ylabel('count')\n",
263 |     "plt.show()\n",
264 |     "\n",
265 |     "# Time progression of fraud rate\n",
266 |     "fraud_rate_by_step = df.groupby('step')[TARGET_COL].mean()\n",
267 |     "fraud_rate_by_step.plot()\n",
268 |     "plt.title('Fraud rate by time step (hour)')\n",
269 |     "plt.xlabel('step')\n",
270 |     "plt.ylabel('fraud_rate')\n",
271 |     "plt.show()\n"
272 |    ]
273 |   },
274 |   {
275 |    "cell_type": "markdown",
276 |    "id": "237fb0ee",
277 |    "metadata": {},
278 |    "source": [
279 |     "## Feature Engineering"
280 |    ]
281 |   },
282 |   {
283 |    "cell_type": "code",
284 |    "execution_count": null,
285 |    "id": "06a4c739",
286 |    "metadata": {},
287 |    "outputs": [],
288 |    "source": [
289 |     "\n",
290 |     "# Derived risk features\n",
291 |     "df['is_type_TRANSFER'] = (df['type'] == 'TRANSFER').astype('int8')\n",
292 |     "df['is_type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype('int8')\n",
293 |     "df['is_merchant_dest'] = df['nameDest'].astype(str).str.startswith('M').astype('int8')\n",
294 |     "\n",
295 |     "# Balance deltas for origin\n",
296 |     "df['deltaOrig'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype('float32')\n",
297 |     "df['deltaDest'] = (df['newbalanceDest'] - df['oldbalanceDest']).astype('float32')\n",
298 |     "\n",
299 |     "# Suspicious patterns\n",
300 |     "df['orig_balance_zero_then_txn'] = ((df['oldbalanceOrg'] == 0) & (df['amount'] > 0)).astype('int8')\n",
301 |     "df['dest_balance_zero_then_in'] = ((df['oldbalanceDest'] == 0) & (df['amount'] > 0)).astype('int8')\n",
302 |     "df['mismatch_orig'] = (np.abs(df['deltaOrig'] - df['amount']) > 1e-2).astype('int8')\n",
303 |     "df['mismatch_dest'] = (np.abs(df['deltaDest'] - df['amount']) > 1e-2).astype('int8')\n",
304 |     "\n",
305 |     "# Drop leakage columns (IDs kept only if used as categorical signals)\n",
306 |     "LEAKS = []  # if any discovered, append here\n",
307 |     "feature_cols = [c for c in df.columns if c not in [TARGET_COL, 'isFlaggedFraud', 'amount_clip'] + LEAKS]\n",
308 |     "print(\"Feature count:\", len(feature_cols))\n",
309 |     "feature_cols[:15]\n"
310 |    ]
311 |   },
312 |   {
313 |    "cell_type": "markdown",
314 |    "id": "c72e8b4f",
315 |    "metadata": {},
316 |    "source": [
317 |     "## Train/Validation Split (Time‑aware)"
318 |    ]
319 |   },
320 |   {
321 |    "cell_type": "code",
322 |    "execution_count": null,
323 |    "id": "fdf30b03",
324 |    "metadata": {},
325 |    "outputs": [],
326 |    "source": [
327 |     "\n",
328 |     "# Time-aware split: use first 80% steps for train, last 20% for validation\n",
329 |     "step_cut = int(df['step'].quantile(0.80))\n",
330 |     "train_idx = df['step'] <= step_cut\n",
331 |     "valid_idx = df['step'] > step_cut\n",
332 |     "\n",
333 |     "train = df.loc[train_idx].reset_index(drop=True)\n",
334 |     "valid = df.loc[valid_idx].reset_index(drop=True)\n",
335 |     "\n",
336 |     "X_train = train[feature_cols]\n",
337 |     "y_train = train[TARGET_COL].astype('int8')\n",
338 |     "X_valid = valid[feature_cols]\n",
339 |     "y_valid = valid[TARGET_COL].astype('int8')\n",
340 |     "\n",
341 |     "print(train.shape, valid.shape, \" | step_cut:\", step_cut)\n",
342 |     "\n",
343 |     "# Encode categoricals (simple): convert category to codes\n",
344 |     "for c in X_train.select_dtypes(include=['category']).columns:\n",
345 |     "    # Ensure consistent codes between train/valid\n",
346 |     "    allcats = pd.Categorical(df[c])\n",
347 |     "    cat2code = {cat: i for i, cat in enumerate(allcats.categories)}\n",
348 |     "    X_train[c] = pd.Categorical(X_train[c], categories=allcats.categories).codes.astype('int32')\n",
349 |     "    X_valid[c] = pd.Categorical(X_valid[c], categories=allcats.categories).codes.astype('int32')\n",
350 |     "\n",
351 |     "# Fill any remaining NaNs (should be rare)\n",
352 |     "X_train = X_train.fillna(0)\n",
353 |     "X_valid = X_valid.fillna(0)\n"
354 |    ]
355 |   },
356 |   {
357 |    "cell_type": "markdown",
358 |    "id": "59afd1e9",
359 |    "metadata": {},
360 |    "source": [
361 |     "## Baseline Models"
362 |    ]
363 |   },
364 |   {
365 |    "cell_type": "code",
366 |    "execution_count": null,
367 |    "id": "44c8ad77",
368 |    "metadata": {},
369 |    "outputs": [],
370 |    "source": [
371 |     "\n",
372 |     "def evaluate_preds(y_true, y_prob, threshold=0.5, title_suffix=\"\"):\n",
373 |     "    y_pred = (y_prob >= threshold).astype(int)\n",
374 |     "    auc = roc_auc_score(y_true, y_prob)\n",
375 |     "    ap = average_precision_score(y_true, y_prob)\n",
376 |     "    cm = confusion_matrix(y_true, y_pred)\n",
377 |     "    print(f\"AUC: {auc:.5f} | Average Precision (PR AUC): {ap:.5f}\")\n",
378 |     "    print(\"Confusion Matrix:\\n\", cm)\n",
379 |     "    print(\"\\nClassification report:\\n\", classification_report(y_true, y_pred, digits=4))\n",
380 |     "\n",
381 |     "    # PR Curve\n",
382 |     "    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)\n",
383 |     "    plt.figure()\n",
384 |     "    plt.plot(recall, precision)\n",
385 |     "    plt.title(f'Precision-Recall Curve {title_suffix}')\n",
386 |     "    plt.xlabel('Recall'); plt.ylabel('Precision')\n",
387 |     "    plt.show()\n",
388 |     "\n",
389 |     "    # ROC Curve\n",
390 |     "    fpr, tpr, _ = roc_curve(y_true, y_prob)\n",
391 |     "    plt.figure()\n",
392 |     "    plt.plot(fpr, tpr)\n",
393 |     "    plt.title(f'ROC Curve {title_suffix}')\n",
394 |     "    plt.xlabel('FPR'); plt.ylabel('TPR')\n",
395 |     "    plt.show()\n",
396 |     "\n",
397 |     "# 8.1 LightGBM (preferred for tabular & imbalance with class_weight)\n",
398 |     "if HAS_LGBM:\n",
399 |     "    lgb_train = lgb.Dataset(X_train, label=y_train)\n",
400 |     "    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)\n",
401 |     "    params = {\n",
402 |     "        'objective': 'binary',\n",
403 |     "        'metric': ['auc'],\n",
404 |     "        'learning_rate': 0.05,\n",
405 |     "        'num_leaves': 64,\n",
406 |     "        'feature_fraction': 0.8,\n",
407 |     "        'bagging_fraction': 0.8,\n",
408 |     "        'bagging_freq': 2,\n",
409 |     "        'reg_lambda': 5.0,\n",
410 |     "        'min_data_in_leaf': 50,\n",
411 |     "        'max_depth': -1,\n",
412 |     "        'verbose': -1,\n",
413 |     "        'scale_pos_weight': max(1.0, (y_train==0).sum() / max(1,(y_train==1).sum())),\n",
414 |     "        'seed': 42\n",
415 |     "    }\n",
416 |     "    print(\"Training LightGBM with params:\", params)\n",
417 |     "    lgb_model = lgb.train(\n",
418 |     "        params,\n",
419 |     "        lgb_train,\n",
420 |     "        valid_sets=[lgb_train, lgb_valid],\n",
421 |     "        valid_names=['train','valid'],\n",
422 |     "        num_boost_round=2000,\n",
423 |     "        early_stopping_rounds=100,\n",
424 |     "        verbose_eval=100\n",
425 |     "    )\n",
426 |     "    y_valid_prob_lgb = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)\n",
427 |     "    evaluate_preds(y_valid, y_valid_prob_lgb, title_suffix=\"(LightGBM)\")\n",
428 |     "else:\n",
429 |     "    print(\"LightGBM not available; skipping.\")\n"
430 |    ]
431 |   },
432 |   {
433 |    "cell_type": "code",
434 |    "execution_count": null,
435 |    "id": "956947ba",
436 |    "metadata": {},
437 |    "outputs": [],
438 |    "source": [
439 |     "\n",
440 |     "if HAS_XGB:\n",
441 |     "    dtrain = xgb.DMatrix(X_train, label=y_train)\n",
442 |     "    dvalid = xgb.DMatrix(X_valid, label=y_valid)\n",
443 |     "    scale_pos_weight = max(1.0, (y_train==0).sum() / max(1,(y_train==1).sum()))\n",
444 |     "    xgb_params = {\n",
445 |     "        'objective': 'binary:logistic',\n",
446 |     "        'eval_metric': 'auc',\n",
447 |     "        'eta': 0.05,\n",
448 |     "        'max_depth': 8,\n",
449 |     "        'subsample': 0.8,\n",
450 |     "        'colsample_bytree': 0.8,\n",
451 |     "        'lambda': 5.0,\n",
452 |     "        'scale_pos_weight': scale_pos_weight,\n",
453 |     "        'seed': 42\n",
454 |     "    }\n",
455 |     "    print(\"Training XGBoost with params:\", xgb_params)\n",
456 |     "    xgb_model = xgb.train(\n",
457 |     "        xgb_params,\n",
458 |     "        dtrain,\n",
459 |     "        num_boost_round=3000,\n",
460 |     "        evals=[(dtrain,'train'),(dvalid,'valid')],\n",
461 |     "        early_stopping_rounds=100,\n",
462 |     "        verbose_eval=200\n",
463 |     "    )\n",
464 |     "    y_valid_prob_xgb = xgb_model.predict(dvalid, iteration_range=(0, xgb_model.best_ntree_limit))\n",
465 |     "    evaluate_preds(y_valid, y_valid_prob_xgb, title_suffix=\"(XGBoost)\")\n",
466 |     "else:\n",
467 |     "    print(\"XGBoost not available; skipping.\")\n"
468 |    ]
469 |   },
470 |   {
471 |    "cell_type": "markdown",
472 |    "id": "ad6486f6",
473 |    "metadata": {},
474 |    "source": [
475 |     "## Explainability — Feature Importance & SHAP"
476 |    ]
477 |   },
478 |   {
479 |    "cell_type": "code",
480 |    "execution_count": null,
481 |    "id": "35f7822e",
482 |    "metadata": {},
483 |    "outputs": [],
484 |    "source": [
485 |     "\n",
486 |     "# Feature importances (model-dependent)\n",
487 |     "def plot_importance(names, importances, topn=25, title=\"Feature Importance\"):\n",
488 |     "    order = np.argsort(importances)[::-1][:topn]\n",
489 |     "    plt.figure(figsize=(8, 6))\n",
490 |     "    plt.barh(range(len(order)), np.array(importances)[order][::-1])\n",
491 |     "    plt.yticks(range(len(order)), np.array(names)[order][::-1])\n",
492 |     "    plt.title(title)\n",
493 |     "    plt.xlabel('importance')\n",
494 |     "    plt.show()\n",
495 |     "\n",
496 |     "if HAS_LGBM:\n",
497 |     "    imp = lgb_model.feature_importance(importance_type='gain')\n",
498 |     "    plot_importance(X_train.columns, imp, title=\"LightGBM Feature Importance (gain)\")\n",
499 |     "\n",
500 |     "# SHAP (optional, can be heavy)\n",
501 |     "if HAS_SHAP and HAS_LGBM:\n",
502 |     "    explainer = shap.TreeExplainer(lgb_model)\n",
503 |     "    # Use a small sample for speed\n",
504 |     "    sample = X_valid.sample(n=min(5000, len(X_valid)), random_state=RANDOM_STATE)\n",
505 |     "    shap_values = explainer.shap_values(sample)\n",
506 |     "    shap.summary_plot(shap_values, sample, show=True)\n"
507 |    ]
508 |   },
509 |   {
510 |    "cell_type": "markdown",
511 |    "id": "341df327",
512 |    "metadata": {},
513 |    "source": [
514 |     "## Multicollinearity (VIF)"
515 |    ]
516 |   },
517 |   {
518 |    "cell_type": "code",
519 |    "execution_count": null,
520 |    "id": "a23982ae",
521 |    "metadata": {},
522 |    "outputs": [],
523 |    "source": [
524 |     "\n",
525 |     "# Compute VIF on numeric subset to check multi-collinearity\n",
526 |     "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
527 |     "import statsmodels.api as sm\n",
528 |     "\n",
529 |     "num_for_vif = X_train.select_dtypes(include=[np.number]).copy()\n",
530 |     "# Limit to a reasonable subset to keep runtime manageable\n",
531 |     "cols_for_vif = [c for c in num_for_vif.columns if num_for_vif[c].nunique() > 10]\n",
532 |     "cols_for_vif = cols_for_vif[:30]  # cap for speed\n",
533 |     "vif_df = pd.DataFrame({\n",
534 |     "    'feature': cols_for_vif,\n",
535 |     "    'VIF': [variance_inflation_factor(num_for_vif[cols_for_vif].values, i) for i in range(len(cols_for_vif))]\n",
536 |     "})\n",
537 |     "vif_df.sort_values('VIF', ascending=False).head(15)\n"
538 |    ]
539 |   },
540 |   {
541 |    "cell_type": "markdown",
542 |    "id": "6b6fbe91",
543 |    "metadata": {},
544 |    "source": [
545 |     "## Threshold Tuning (Optimize for Business Cost)"
546 |    ]
547 |   },
548 |   {
549 |    "cell_type": "code",
550 |    "execution_count": null,
551 |    "id": "26c01931",
552 |    "metadata": {},
553 |    "outputs": [],
554 |    "source": [
555 |     "\n",
556 |     "# Example: choose threshold maximizing F1 or desired precision\n",
557 |     "def best_threshold_by_precision_target(y_true, y_prob, precision_target=0.95):\n",
558 |     "    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)\n",
559 |     "    for p, r, t in zip(precision, recall, np.append(thresholds, 1)):\n",
560 |     "        if p >= precision_target:\n",
561 |     "            return float(t), float(p), float(r)\n",
562 |     "    return 0.5, precision[0], recall[0]\n",
563 |     "\n",
564 |     "if HAS_LGBM:\n",
565 |     "    thr, p, r = best_threshold_by_precision_target(y_valid, y_valid_prob_lgb, precision_target=0.90)\n",
566 |     "    print(f\"Threshold for >=90% precision: {thr:.4f} -> precision={p:.3f}, recall={r:.3f}\")\n"
567 |    ]
568 |   },
569 |   {
570 |    "cell_type": "markdown",
571 |    "id": "79ab0503",
572 |    "metadata": {},
573 |    "source": [
574 |     "## Save Artifacts"
575 |    ]
576 |   },
577 |   {
578 |    "cell_type": "code",
579 |    "execution_count": null,
580 |    "id": "27e32027",
581 |    "metadata": {},
582 |    "outputs": [],
583 |    "source": [
584 |     "\n",
585 |     "# Save predictions for audit / attachment in submission\n",
586 |     "if HAS_LGBM:\n",
587 |     "    valid_out = valid[['step','type','amount','nameOrig','nameDest','isFlaggedFraud',TARGET_COL]].copy()\n",
588 |     "    valid_out['fraud_prob_lgb'] = y_valid_prob_lgb\n",
589 |     "    valid_out.to_parquet('valid_predictions.parquet', index=False)\n",
590 |     "    print(\"Saved: valid_predictions.parquet\")\n"
591 |    ]
592 |   },
593 |   {
594 |    "cell_type": "markdown",
595 |    "id": "4604c2e3",
596 |    "metadata": {},
597 |    "source": [
598 |     "\n",
599 |     "## Conclusions & Answers (fill after running)\n",
600 |     "\n",
601 |     "**1) Data cleaning:**  \n",
602 |     "- Missing: _<notes>_  \n",
603 |     "- Outliers: _<notes>_  \n",
604 |     "- Multicollinearity: _<notes>_  \n",
605 |     "\n",
606 |     "**2) Model description:**  \n",
607 |     "- Algorithm: _<LightGBM/XGBoost>_  \n",
608 |     "- Why: _<reasoning>_  \n",
609 |     "- Handling imbalance: _<scale_pos_weight, threshold tuning>_  \n",
610 |     "\n",
611 |     "**3) Variable selection:**  \n",
612 |     "- Included: engineered deltas, type indicators, merchant flags, mismatch signals  \n",
613 |     "- Rationale: information value, importance, SHAP  \n",
614 |     "\n",
615 |     "**4) Performance:**  \n",
616 |     "- Metrics: AUC, PR‑AUC, Precision@Recall, confusion matrix  \n",
617 |     "- Validation: time‑aware split (first 80% steps train, last 20% validate)  \n",
618 |     "\n",
619 |     "**5) Key predictors:**  \n",
620 |     "- Top features: _<from gain importance / SHAP>_  \n",
621 |     "\n",
622 |     "**6) Do they make sense?**  \n",
623 |     "- Business reasoning: _<explain link to fraud modus operandi>_  \n",
624 |     "\n",
625 |     "**7) Prevention recommendations:**  \n",
626 |     "- _<rate limiting high‑risk flows, MFA, velocity rules, beneficiary cooling periods, anomaly detection in balances, merchant monitoring, graph rules, device fingerprinting>_  \n",
627 |     "\n",
628 |     "**8) Measurement of success:**  \n",
629 |     "- A/B or backtest: reduction in fraud loss, false positive rate, precision/recall lift, alert fatigue drop, manual review SLA, net profit impact.\n"
630 |    ]
631 |   }
632 |  ],
633 |  "metadata": {
634 |   "kernelspec": {
635 |    "display_name": "fraud-env",
636 |    "language": "python",
637 |    "name": "python3"
638 |   },
639 |   "language_info": {
640 |    "codemirror_mode": {
641 |     "name": "ipython",
642 |     "version": 3
643 |    },
644 |    "file_extension": ".py",
645 |    "mimetype": "text/x-python",
646 |    "name": "python",
647 |    "nbconvert_exporter": "python",
648 |    "pygments_lexer": "ipython3",
649 |    "version": "3.13.3"
650 |   }
651 |  },
652 |  "nbformat": 4,
653 |  "nbformat_minor": 5
654 | }
655 | 


--------------------------------------------------------------------------------
/Executive_Summary_Template.md:
--------------------------------------------------------------------------------
 1 | 
 2 | # Fraud Detection — Executive Summary (Submission)
 3 | 
 4 | **Problem**  
 5 | Build a proactive fraud detection model for a financial company using 6.36M transactions (10 columns).
 6 | 
 7 | **Approach**  
 8 | - Time‑aware split (train: first 80% hours; validate: last 20%).  
 9 | - Memory‑efficient load with typed columns.  
10 | - Features: transaction type flags, origin/destination balance deltas, mismatch signals, merchant flag, velocity by account (optional extension).  
11 | - Models: LightGBM primary, XGBoost fallback; class weighting + threshold tuning to business precision.  
12 | - Explainability: feature importance + SHAP (sampled).
13 | 
14 | **Data Quality**  
15 | - Missing values: _<fill>_  
16 | - Outliers: handled by robust models & clipping for EDA; no leakage.  
17 | - Multicollinearity: VIF checked on numeric subset; _<notes>_.
18 | 
19 | **Results**  
20 | - ROC‑AUC: _<value>_  
21 | - PR‑AUC: _<value>_  
22 | - Precision @ _<Recall>_: _<value>_  
23 | - Key drivers: _<features>_.
24 | 
25 | **Business Interpretation**  
26 | - Factors align with known fraud patterns (fast drain via TRANSFER→CASH_OUT, zero‑balance anomalies, merchant recipients).
27 | 
28 | **Recommendations**  
29 | 1. High‑value transfer limits with step‑up auth.  
30 | 2. Velocity rules by account/beneficiary; cool‑off windows.  
31 | 3. Anomaly monitoring on `deltaOrig`, `deltaDest` vs history.  
32 | 4. Device/IP fingerprinting + geo‑velocity checks.  
33 | 5. Graph analytics for mule detection; merchant vetting.  
34 | 6. Human‑in‑the‑loop triage and feedback loop to model.
35 | 
36 | **Success Metrics & Monitoring**  
37 | - Weekly PR‑AUC, Precision@k, false positive rate, alert SLA, prevented loss (₹), and net profit impact.  
38 | - Shadow/A‑B test against current rules; rollout with canary + guardrails.
39 | 
40 | 
41 | 

