from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.pkl",
    "XGBoost Pipeline": MODEL_DIR / "xgboost_fraud_detection.pkl",
}

FEATURE_COLUMNS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOLEAN_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

BOOLEAN_VALUES = {
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
}


st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .app-title { font-size: 2rem; font-weight: 700; margin: 0; }
    .app-subtitle { font-size: 0.95rem; color: #94a3b8; margin-top: 0.4rem; }

    .risk-high-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.25rem;
        border-radius: 8px;
        color: #991b1b;
        font-weight: 600;
    }
    .risk-med-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 8px;
        color: #92400e;
        font-weight: 600;
    }
    .risk-low-box {
        background-color: #f0fdf4;
        border-left: 5px solid #10b981;
        padding: 1.25rem;
        border-radius: 8px;
        color: #065f46;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    loaded_models = {}
    load_errors = {}

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            continue

        try:
            loaded_models[model_name] = joblib.load(model_path)
        except Exception as exc:
            load_errors[model_name] = str(exc)

    return loaded_models, load_errors


def prepare_model_input(dataframe):
    """Keep the model inputs in the same format used during training."""
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    model_input = dataframe.loc[:, FEATURE_COLUMNS].copy()

    for column in BOOLEAN_COLUMNS:
        values = model_input[column]
        if pd.api.types.is_bool_dtype(values):
            model_input[column] = values.astype(int)
        else:
            converted = values.astype(str).str.strip().str.lower().map(BOOLEAN_VALUES)
            if converted.isna().any():
                raise ValueError(
                    f"Column '{column}' must contain True/False or 1/0 values."
                )
            model_input[column] = converted.astype(int)

    return model_input


loaded_models, load_errors = load_artifacts()


st.markdown(
    """
    <div class="main-header">
        <p class="app-title">Fraud Detection System</p>
        <p class="app-subtitle">Machine Learning model evaluation and transaction risk analysis dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if load_errors:
    for model_name, error_message in load_errors.items():
        st.warning(
            f"{model_name} could not be loaded. Check the saved model and "
            f"the installed package versions. Details: {error_message}"
        )

if not loaded_models:
    st.error(
        "No model files are available. Place at least one trained model in the "
        "`models` folder. The expected files are "
        "`logistic_regression_model.pkl` and `xgboost_fraud_detection.pkl`."
    )
    st.stop()


with st.sidebar:
    st.markdown("### Model Settings")
    model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

active_model = loaded_models[model_choice]


tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Transaction",
        "📁 Batch CSV Scoring",
        "📊 Model Info",
    ]
)


with tab1:
    st.markdown("#### Test a Transaction")
    st.markdown(
        "Fill in the transaction details below or use the quick-fill buttons to "
        "test sample data."
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        load_normal = st.button("📥 Load Sample Normal Data")
    with col_btn2:
        load_fraud = st.button("📥 Load Sample Fraud Data")

    if "sample_vals" not in st.session_state:
        st.session_state.sample_vals = {
            "amount_usd": 75.0,
            "account_balance_usd": 2500.0,
            "velocity_score": 10.0,
            "txn_count_last_24h": 2,
            "hours_since_last_txn": 5.0,
            "distance_from_home_km": 5.0,
            "customer_age": 32,
            "card_age_months": 36,
            "cvv_retry_count": 0,
            "merchant_risk_score": 15.0,
            "prior_disputes": 0,
            "time_of_day_hour": 14,
            "day_of_week": 2,
            "merchant_category": "Retail",
            "card_type": "Debit",
            "auth_method": "OTP",
            "channel": "Online",
            "device_type": "iPhone",
            "is_foreign_transaction": False,
            "is_new_merchant": False,
            "used_vpn": False,
            "ip_country_mismatch": False,
            "billing_shipping_mismatch": False,
            "is_ai_generated_scam_attempt": False,
        }

    if load_normal:
        st.session_state.sample_vals = {
            "amount_usd": 45.00,
            "account_balance_usd": 4200.00,
            "velocity_score": 5.0,
            "txn_count_last_24h": 1,
            "hours_since_last_txn": 12.0,
            "distance_from_home_km": 3.2,
            "customer_age": 29,
            "card_age_months": 48,
            "cvv_retry_count": 0,
            "merchant_risk_score": 10.0,
            "prior_disputes": 0,
            "time_of_day_hour": 12,
            "day_of_week": 1,
            "merchant_category": "Grocery",
            "card_type": "Debit",
            "auth_method": "PIN",
            "channel": "POS",
            "device_type": "iPhone",
            "is_foreign_transaction": False,
            "is_new_merchant": False,
            "used_vpn": False,
            "ip_country_mismatch": False,
            "billing_shipping_mismatch": False,
            "is_ai_generated_scam_attempt": False,
        }
        st.rerun()

    if load_fraud:
        st.session_state.sample_vals = {
            "amount_usd": 980.50,
            "account_balance_usd": 120.00,
            "velocity_score": 88.5,
            "txn_count_last_24h": 14,
            "hours_since_last_txn": 0.1,
            "distance_from_home_km": 1250.4,
            "customer_age": 45,
            "card_age_months": 2,
            "cvv_retry_count": 3,
            "merchant_risk_score": 92.0,
            "prior_disputes": 2,
            "time_of_day_hour": 3,
            "day_of_week": 6,
            "merchant_category": "Electronics",
            "card_type": "Credit",
            "auth_method": "No Authentication",
            "channel": "Online",
            "device_type": "Windows PC",
            "is_foreign_transaction": True,
            "is_new_merchant": True,
            "used_vpn": True,
            "ip_country_mismatch": True,
            "billing_shipping_mismatch": True,
            "is_ai_generated_scam_attempt": True,
        }
        st.rerun()

    sv = st.session_state.sample_vals

    with st.form("fraud_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Financial & Velocity**")
            amount_usd = st.number_input(
                "Transaction Amount (USD)",
                min_value=0.0,
                value=float(sv["amount_usd"]),
            )
            account_balance_usd = st.number_input(
                "Account Balance (USD)",
                min_value=0.0,
                value=float(sv["account_balance_usd"]),
            )
            velocity_score = st.number_input(
                "Velocity Score",
                min_value=0.0,
                max_value=100.0,
                value=float(sv["velocity_score"]),
            )
            txn_count_last_24h = st.number_input(
                "Txn Count (Last 24h)",
                min_value=0,
                max_value=50,
                value=int(sv["txn_count_last_24h"]),
            )
            hours_since_last_txn = st.number_input(
                "Hours Since Last Txn",
                min_value=0.0,
                value=float(sv["hours_since_last_txn"]),
            )

        with col2:
            st.markdown("**Behavioral Details**")
            distance_from_home_km = st.number_input(
                "Distance From Home (km)",
                min_value=0.0,
                value=float(sv["distance_from_home_km"]),
            )
            customer_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=int(sv["customer_age"]),
            )
            card_age_months = st.number_input(
                "Card Age (Months)",
                min_value=0,
                max_value=240,
                value=int(sv["card_age_months"]),
            )
            cvv_retry_count = st.selectbox(
                "CVV Retry Count",
                options=[0, 1, 2, 3],
                index=int(sv["cvv_retry_count"]),
            )
            merchant_risk_score = st.slider(
                "Merchant Risk Score",
                0.0,
                100.0,
                float(sv["merchant_risk_score"]),
            )
            prior_disputes = st.selectbox(
                "Prior Disputes",
                options=[0, 1, 2, 3, 4],
                index=int(sv["prior_disputes"]),
            )

        with col3:
            st.markdown("**Categories & Metadata**")
            time_of_day_hour = st.slider(
                "Time of Day (Hour)", 0, 23, int(sv["time_of_day_hour"])
            )
            day_of_week = st.selectbox(
                "Day of Week (0=Mon)",
                options=[0, 1, 2, 3, 4, 5, 6],
                index=int(sv["day_of_week"]),
            )

            cat_options = ["Retail", "Electronics", "Travel", "Grocery", "Digital"]
            merchant_category = st.selectbox(
                "Merchant Category",
                options=cat_options,
                index=(
                    cat_options.index(sv["merchant_category"])
                    if sv["merchant_category"] in cat_options
                    else 0
                ),
            )

            card_options = ["Debit", "Credit", "Prepaid"]
            card_type = st.selectbox(
                "Card Type",
                options=card_options,
                index=(
                    card_options.index(sv["card_type"])
                    if sv["card_type"] in card_options
                    else 0
                ),
            )

            auth_options = [
                "OTP",
                "PIN",
                "3D Secure",
                "Biometric",
                "No Authentication",
            ]
            auth_method = st.selectbox(
                "Auth Method",
                options=auth_options,
                index=(
                    auth_options.index(sv["auth_method"])
                    if sv["auth_method"] in auth_options
                    else 0
                ),
            )

            channel_options = ["Online", "POS", "In-App", "ATM"]
            channel = st.selectbox(
                "Channel",
                options=channel_options,
                index=(
                    channel_options.index(sv["channel"])
                    if sv["channel"] in channel_options
                    else 0
                ),
            )

            device_options = ["iPhone", "Android Phone", "Windows PC", "Tablet"]
            device_type = st.selectbox(
                "Device Type",
                options=device_options,
                index=(
                    device_options.index(sv["device_type"])
                    if sv["device_type"] in device_options
                    else 0
                ),
            )

        st.markdown("---")
        st.markdown("**Risk Flags**")
        f1, f2, f3, f4, f5, f6 = st.columns(6)

        with f1:
            is_foreign_transaction = st.checkbox(
                "Foreign Txn", value=bool(sv["is_foreign_transaction"])
            )
        with f2:
            is_new_merchant = st.checkbox(
                "New Merchant", value=bool(sv["is_new_merchant"])
            )
        with f3:
            used_vpn = st.checkbox("Used VPN", value=bool(sv["used_vpn"]))
        with f4:
            ip_country_mismatch = st.checkbox(
                "IP Mismatch", value=bool(sv["ip_country_mismatch"])
            )
        with f5:
            billing_shipping_mismatch = st.checkbox(
                "Billing Mismatch", value=bool(sv["billing_shipping_mismatch"])
            )
        with f6:
            is_ai_generated_scam_attempt = st.checkbox(
                "AI Scam Attempt", value=bool(sv["is_ai_generated_scam_attempt"])
            )

        submitted = st.form_submit_button(
            "Run Prediction", type="primary", use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "amount_usd": amount_usd,
                    "hours_since_last_txn": hours_since_last_txn,
                    "txn_count_last_24h": txn_count_last_24h,
                    "distance_from_home_km": distance_from_home_km,
                    "card_age_months": card_age_months,
                    "customer_age": customer_age,
                    "account_balance_usd": account_balance_usd,
                    "cvv_retry_count": cvv_retry_count,
                    "velocity_score": velocity_score,
                    "time_of_day_hour": time_of_day_hour,
                    "day_of_week": day_of_week,
                    "merchant_risk_score": merchant_risk_score,
                    "prior_disputes": prior_disputes,
                    "merchant_category": merchant_category,
                    "card_type": card_type,
                    "auth_method": auth_method,
                    "channel": channel,
                    "device_type": device_type,
                    "is_foreign_transaction": is_foreign_transaction,
                    "is_new_merchant": is_new_merchant,
                    "used_vpn": used_vpn,
                    "ip_country_mismatch": ip_country_mismatch,
                    "billing_shipping_mismatch": billing_shipping_mismatch,
                    "is_ai_generated_scam_attempt": is_ai_generated_scam_attempt,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        try:
            model_input = prepare_model_input(input_data)
            prediction_prob = active_model.predict_proba(model_input)[0][1]

            st.markdown("---")
            st.markdown("#### Prediction Results")

            res_col1, res_col2 = st.columns([1, 2], gap="large")

            with res_col1:
                st.metric(
                    label="Fraud Probability",
                    value=f"{prediction_prob * 100:.2f}%",
                )

            with res_col2:
                if prediction_prob >= 0.7:
                    st.markdown(
                        '<div class="risk-high-box">⚠️ HIGH RISK TRANSACTION<br><span'
                        ' style="font-size: 0.85rem; font-weight: 400;">The model'
                        " flags this transaction as high risk for fraud.</span></div>",
                        unsafe_allow_html=True,
                    )
                elif prediction_prob >= 0.3:
                    st.markdown(
                        '<div class="risk-med-box">⚡ MEDIUM RISK TRANSACTION<br><span'
                        ' style="font-size: 0.85rem; font-weight: 400;">Moderate'
                        " anomaly detected. Verification recommended.</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="risk-low-box">✅ LOW RISK TRANSACTION<br><span'
                        ' style="font-size: 0.85rem; font-weight: 400;">Transaction'
                        " appears legitimate.</span></div>",
                        unsafe_allow_html=True,
                    )

        except Exception as exc:
            st.error(f"Error during prediction: {exc}")


with tab2:
    st.markdown("#### Batch CSV Scoring")
    st.markdown(
        "Upload a CSV file containing your test dataset to evaluate multiple "
        "rows at once."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if batch_df.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                st.success(f"Loaded {len(batch_df)} rows successfully.")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Run Batch Scoring"):
                    model_input = prepare_model_input(batch_df)
                    batch_probs = active_model.predict_proba(model_input)[:, 1]

                    results_df = batch_df.copy()
                    results_df["fraud_probability"] = batch_probs
                    results_df["risk_level"] = pd.cut(
                        batch_probs,
                        bins=[-0.1, 0.3, 0.7, 1.0],
                        labels=["Low Risk", "Medium Risk", "High Risk"],
                    )

                    st.dataframe(results_df, use_container_width=True)

                    csv_export = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_export,
                        file_name="fraud_predictions_output.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Batch prediction error: {exc}")


with tab3:
    st.markdown("#### Model Overview")
    st.write(
        "- **XGBoost Pipeline:** Handles non-linear relationships and feature "
        "interactions efficiently.\n"
        "- **Logistic Regression:** Serves as a linear baseline model for "
        "comparison."
    )
    st.write(f"Input features used by the models: {len(FEATURE_COLUMNS)}")
    st.caption(
        "The Low, Medium and High Risk labels use dashboard thresholds of "
        "30% and 70%. These are display thresholds, not a validated decision policy."
    )
