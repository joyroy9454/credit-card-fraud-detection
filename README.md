# Fraud Detection System

A machine learning project for identifying potentially fraudulent credit-card transactions. It includes a Jupyter notebook for data analysis and model training, plus a Streamlit dashboard for testing individual transactions and scoring CSV files.

## Dataset and Kaggle competition

This project uses the dataset provided for **[Predicting Credit Card Fraud — Kaggle Playground Series, Season 6 Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11)**.

The competition asks participants to predict the probability that a transaction is fraudulent (`is_fraud`). Kaggle evaluates submissions using ROC-AUC. Kaggle describes its Playground Series datasets as synthetically generated for machine-learning practice; this project does not claim to have collected or created the original dataset.

To obtain the data, open the competition page, sign in to Kaggle, accept the competition rules, and download the files from its **Data** tab. Place the training file at:

```text
data/raw/train.csv
```

The raw competition files are intentionally excluded from Git by `.gitignore`. Do not upload or redistribute the data through this repository unless the competition's current rules and applicable dataset terms explicitly allow it. Other users should obtain the files directly from Kaggle and follow the access requirements there.

**Competition citation**

Yao Yan, Walter Reade, and Elizabeth Park. *Predicting Credit Card Fraud*. Kaggle, 2026. https://www.kaggle.com/competitions/playground-series-s6e11.

## What is included

- Exploratory analysis of the transaction data.
- Data preprocessing with `StandardScaler` and `OneHotEncoder`.
- A Logistic Regression baseline.
- An XGBoost pipeline with class-imbalance adjustment and randomized hyperparameter search.
- A Streamlit app with single-transaction prediction and batch CSV scoring.

## Project layout

```text
fraud-detection/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── fraud_detection.ipynb
├── models/
│   ├── logistic_regression_model.pkl
│   └── xgboost_fraud_detection.pkl
└── data/
    ├── raw/
    │   └── train.csv          # Download from the Kaggle competition
    └── processed/
```

The starter repository does not include the competition data or trained model binaries. Add the dataset locally and generate the model files by running the notebook.

## Run the dashboard

Use Python 3.12 if possible. From the project folder, create an environment and install the packages:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the requirements and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Place at least one of the trained model files in `models/` before starting the dashboard:

- `models/logistic_regression_model.pkl`
- `models/xgboost_fraud_detection.pkl`

The app can load either model independently. If both files are present, you can switch between them in the sidebar.

## Train the models again

Download the competition's training data and place it at:

```text
data/raw/train.csv
```

Then open the notebook from the project folder:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

Run the cells in order. The notebook saves the trained pipelines to `models/` using the filenames listed above.

## Batch CSV format

The batch-scoring tab expects the same 24 input features used during training. The CSV should contain these columns:

```text
amount_usd
hours_since_last_txn
txn_count_last_24h
distance_from_home_km
card_age_months
customer_age
account_balance_usd
cvv_retry_count
velocity_score
time_of_day_hour
day_of_week
merchant_risk_score
prior_disputes
is_foreign_transaction
is_new_merchant
used_vpn
ip_country_mismatch
billing_shipping_mismatch
is_ai_generated_scam_attempt
merchant_category
card_type
auth_method
channel
device_type
```

Do not include `id` or `is_fraud` as model input columns. Boolean columns should use `True`/`False` or `1`/`0`. The app checks for missing feature columns before scoring.

## Model results recorded in the notebook

The saved notebook output reports the following baseline results:

| Model | ROC-AUC | PR-AUC | Average Precision |
|---|---:|---:|---:|
| Logistic Regression | 0.9404 | 0.4969 | 0.4970 |
| XGBoost (before tuning) | 0.9406 | 0.4860 | 0.4861 |

The notebook also runs randomized hyperparameter search for XGBoost using average precision. The final tuned-model evaluation cell does not have saved output in the uploaded notebook, so no tuned score is listed here.

The target column has an approximate fraud rate of 1.67% in the dataset summary. For this reason, precision-recall metrics are included alongside ROC-AUC. The dashboard's 30% and 70% risk bands are display thresholds in the app; they have not been established as a production decision policy.

