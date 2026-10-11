import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.decomposition import PCA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial AI Command Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #9ca3af;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
}

.metric-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
}

.risk-high {
    padding: 20px;
    border-radius: 12px;
    background: rgba(255, 70, 70, 0.15);
    border: 1px solid rgba(255, 70, 70, 0.40);
}

.risk-normal {
    padding: 20px;
    border-radius: 12px;
    background: rgba(0, 200, 100, 0.12);
    border: 1px solid rgba(0, 200, 100, 0.30);
}

.small-text {
    color: #9ca3af;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = "dataset/creditcard.csv"

MODEL_PATH = "models/isolation_forest.pkl"
LOF_PATH = "models/lof.pkl"

AMOUNT_SCALER_PATH = "models/amount_scaler.pkl"
TIME_SCALER_PATH = "models/time_scaler.pkl"

THRESHOLD_PATH = "models/isolation_forest_threshold.pkl"


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_ORDER = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_models():

    isolation_forest = joblib.load(MODEL_PATH)

    lof = joblib.load(LOF_PATH)

    amount_scaler = joblib.load(
        AMOUNT_SCALER_PATH
    )

    time_scaler = joblib.load(
        TIME_SCALER_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    return (
        isolation_forest,
        lof,
        amount_scaler,
        time_scaler,
        threshold
    )


# ============================================================
# INITIALIZE
# ============================================================

df = load_dataset()

(
    isolation_forest,
    lof,
    amount_scaler,
    time_scaler,
    saved_threshold
) = load_models()


# ============================================================
# PREPARE FEATURES
# ============================================================

@st.cache_data
def prepare_features(data):

    X = data.drop(
        "Class",
        axis=1
    ).copy()

    X["Amount"] = amount_scaler.transform(
        X[["Amount"]]
    )

    X["Time"] = time_scaler.transform(
        X[["Time"]]
    )

    X = X[FEATURE_ORDER]

    return X


X = prepare_features(df)

y = df["Class"].values


# ============================================================
# ANOMALY SCORES
# ============================================================

@st.cache_data
def calculate_all_scores(_model, X_data):

    return -_model.decision_function(
        X_data
    )


all_scores = calculate_all_scores(
    isolation_forest,
    X
)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_transactions = len(df)

normal_transactions = int(
    (df["Class"] == 0).sum()
)

fraud_transactions = int(
    (df["Class"] == 1).sum()
)

fraud_rate = (
    fraud_transactions /
    total_transactions
) * 100


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 Financial AI")

    st.caption(
        "Anomaly Detection & Investigation Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Transaction Investigation",
            "🚨 Fraud Detection",
            "🎚️ Threshold Analysis",
            "🤖 Model Comparison",
            "📊 Dataset Analytics",
            "🔬 PCA Visualization",
            "📋 Detection History"
        ]
    )

    st.divider()

    st.success(
        "🟢 AI System Active"
    )

    st.metric(
        "Detection Threshold",
        f"{saved_threshold:.6f}"
    )

    st.caption(
        "Isolation Forest"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 Financial AI Command Center</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Financial Anomaly Detection & Investigation Platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Executive Overview")

    st.write(
        "AI-powered monitoring system for identifying unusual "
        "financial transaction behavior."
    )

    st.divider()

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Normal Transactions",
            f"{normal_transactions:,}"
        )

    with col3:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    with col4:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.3f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("🤖 Current Model Performance")

    final_predictions = (
        all_scores >= saved_threshold
    ).astype(int)

    precision = precision_score(
        y,
        final_predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        final_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        final_predictions,
        zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col3:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Threshold",
            f"{saved_threshold:.6f}"
        )

    st.divider()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Transaction Distribution")

        class_counts = pd.DataFrame({
            "Class": [
                "Normal",
                "Fraud"
            ],
            "Transactions": [
                normal_transactions,
                fraud_transactions
            ]
        })

        st.bar_chart(
            class_counts.set_index("Class")
        )

    with col2:

        st.subheader("🚨 Detection Summary")

        cm = confusion_matrix(
            y,
            final_predictions
        )

        tn, fp, fn, tp = cm.ravel()

        summary = pd.DataFrame({
            "Metric": [
                "True Negatives",
                "False Positives",
                "False Negatives",
                "True Positives"
            ],
            "Count": [
                tn,
                fp,
                fn,
                tp
            ]
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    st.info(
        """
        🧠 **System Pipeline**

        Transaction → Preprocessing → Isolation Forest →
        Anomaly Score → Threshold Engine → Risk Decision
        """
    )


# ============================================================
# PAGE 2 - TRANSACTION INVESTIGATION
# ============================================================

elif page == "🔍 Transaction Investigation":

    st.header("🔍 Transaction Investigation")

    st.write(
        "Select any transaction from the dataset and perform "
        "a detailed anomaly analysis."
    )

    st.divider()

    transaction_type = st.selectbox(
        "Select Transaction Type",
        [
            "Normal Transactions",
            "Fraud Transactions",
            "All Transactions"
        ]
    )

    if transaction_type == "Normal Transactions":

        available_df = df[
            df["Class"] == 0
        ].reset_index()

    elif transaction_type == "Fraud Transactions":

        available_df = df[
            df["Class"] == 1
        ].reset_index()

    else:

        available_df = df.reset_index()


    transaction_number = st.number_input(
        "Transaction Number",
        min_value=0,
        max_value=len(available_df) - 1,
        value=0,
        step=1
    )

    selected = available_df.iloc[
        transaction_number
    ]

    original_index = selected["index"]

    st.divider()

    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    st.subheader("📄 Transaction Details")

    actual_class = int(
        selected["Class"]
    )

    actual_label = (
        "FRAUD"
        if actual_class == 1
        else "NORMAL"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Transaction ID",
            str(original_index)
        )

    with col2:

        st.metric(
            "Transaction Amount",
            f"${selected['Amount']:.2f}"
        )

    with col3:

        st.metric(
            "Transaction Time",
            f"{selected['Time']:.2f}"
        )

    with col4:

        st.metric(
            "Actual Class",
            actual_label
        )

    st.divider()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if st.button(
        "🚨 Analyze Transaction",
        use_container_width=True
    ):

        selected_features = selected.drop(
            "Class"
        ).copy()

        transaction_X = pd.DataFrame(
            [selected_features]
        )

        transaction_X["Amount"] = amount_scaler.transform(
            transaction_X[["Amount"]]
        )

        transaction_X["Time"] = time_scaler.transform(
            transaction_X[["Time"]]
        )

        transaction_X = transaction_X[
            FEATURE_ORDER
        ]

        anomaly_score = -isolation_forest.decision_function(
            transaction_X
        )[0]

        predicted_class = int(
            anomaly_score >= saved_threshold
        )

        prediction = (
            "ANOMALY / POSSIBLE FRAUD"
            if predicted_class == 1
            else "NORMAL"
        )

        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        if anomaly_score <= 0:

            risk_score = 0

        else:

            risk_score = min(
                100,
                (anomaly_score / saved_threshold) * 70
            )

        if risk_score >= 80:

            risk_level = "CRITICAL"

        elif risk_score >= 50:

            risk_level = "HIGH"

        elif risk_score >= 25:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        st.session_state.history.append({
            "Transaction ID": int(original_index),
            "Amount": float(selected["Amount"]),
            "Actual": actual_label,
            "Prediction": prediction,
            "Anomaly Score": float(anomaly_score),
            "Risk Score": float(risk_score),
            "Risk Level": risk_level
        })


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("🧠 AI Detection Result")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "AI Prediction",
                prediction
            )

        with col2:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.6f}"
            )

        with col3:

            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/100"
            )

        with col4:

            st.metric(
                "Risk Level",
                risk_level
            )


        if predicted_class == 1:

            st.error(
                "🚨 ANOMALY DETECTED / POSSIBLE FRAUD"
            )

        else:

            st.success(
                "✅ NORMAL TRANSACTION"
            )


        if predicted_class == actual_class:

            st.success(
                "✅ Model prediction matches the actual class."
            )

        else:

            st.warning(
                "⚠️ Model prediction does not match the actual class."
            )


        # ----------------------------------------------------
        # SCORE VS THRESHOLD
        # ----------------------------------------------------

        st.subheader("🎚️ Score vs Detection Threshold")

        score_df = pd.DataFrame({
            "Value": [
                anomaly_score,
                saved_threshold
            ]
        }, index=[
            "Anomaly Score",
            "Threshold"
        ])

        st.bar_chart(score_df)


        # ----------------------------------------------------
        # TOP FEATURE DEVIATIONS
        # ----------------------------------------------------

        st.subheader("🔬 Feature Deviation Analysis")

        scaled_transaction = transaction_X.iloc[0]

        feature_names = [
            f"V{i}"
            for i in range(1, 29)
        ]

        feature_values = []

        for feature in feature_names:

            feature_values.append({
                "Feature": feature,
                "Scaled Value": abs(
                    float(scaled_transaction[feature])
                )
            })

        deviation_df = pd.DataFrame(
            feature_values
        ).sort_values(
            "Scaled Value",
            ascending=False
        ).head(10)

        st.bar_chart(
            deviation_df.set_index(
                "Feature"
            )
        )

        st.caption(
            "This chart shows features with the largest absolute "
            "scaled values. It is a deviation view, not a causal "
            "explanation of fraud."
        )


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        with st.expander(
            "📋 View Complete Transaction Features"
        ):

            st.dataframe(
                selected.drop(
                    "Class"
                ).to_frame("Value"),
                use_container_width=True
            )


# ============================================================
# PAGE 3 - FRAUD DETECTION
# ============================================================

elif page == "🚨 Fraud Detection":

    st.header("🚨 Fraud Detection Center")

    st.write(
        "Detailed overview of fraudulent transactions and "
        "Isolation Forest detection performance."
    )

    st.divider()

    predictions = (
        all_scores >= saved_threshold
    ).astype(int)

    cm = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    detection_rate = (
        tp / fraud_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Actual Fraud",
            f"{fraud_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Detected",
            f"{tp:,}"
        )

    with col3:

        st.metric(
            "Fraud Missed",
            f"{fn:,}"
        )

    with col4:

        st.metric(
            "Detection Rate",
            f"{detection_rate:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🔢 Isolation Forest Confusion Matrix"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="Blues",
        linewidths=1,
        xticklabels=[
            "Predicted Normal",
            "Predicted Fraud"
        ],
        yticklabels=[
            "Actual Normal",
            "Actual Fraud"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "Isolation Forest Confusion Matrix"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # CONFUSION MATRIX DETAILS
    # --------------------------------------------------------

    st.subheader(
        "📖 Confusion Matrix Explanation"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
            **True Negative (TN)**

            {tn:,} normal transactions were correctly
            identified as normal.
            """
        )

        st.warning(
            f"""
            **False Positive (FP)**

            {fp:,} normal transactions were incorrectly
            flagged as anomalies.
            """
        )

    with col2:

        st.error(
            f"""
            **False Negative (FN)**

            {fn:,} fraudulent transactions were missed
            by the model.
            """
        )

        st.success(
            f"""
            **True Positive (TP)**

            {tp:,} fraudulent transactions were correctly
            detected.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL FRAUD TABLE
    # --------------------------------------------------------

    st.subheader(
        "🔴 Fraud Transactions"
    )

    fraud_display = df[
        df["Class"] == 1
    ].copy()

    fraud_display["Anomaly Score"] = all_scores[
        df["Class"].values == 1
    ]

    fraud_display["Model Decision"] = np.where(
        fraud_display["Anomaly Score"] >= saved_threshold,
        "DETECTED",
        "MISSED"
    )

    fraud_display["Risk Score"] = np.minimum(
        100,
        np.maximum(
            0,
            (fraud_display["Anomaly Score"] /
             saved_threshold) * 70
        )
    )

    display_columns = [
        "Time",
        "Amount",
        "Anomaly Score",
        "Risk Score",
        "Model Decision"
    ]

    st.dataframe(
        fraud_display[
            display_columns
        ].sort_values(
            "Anomaly Score",
            ascending=False
        ).head(100),
        use_container_width=True
    )


# ============================================================
# PAGE 4 - THRESHOLD ANALYSIS
# ============================================================

elif page == "🎚️ Threshold Analysis":

    st.header("🎚️ Threshold Analysis Lab")

    st.write(
        "Explore how the anomaly threshold changes "
        "Precision, Recall and F1 Score."
    )

    st.divider()

    threshold_value = st.slider(
        "Detection Threshold",
        min_value=0.001,
        max_value=0.150,
        value=float(saved_threshold),
        step=0.001,
        format="%.3f"
    )

    threshold_predictions = (
        all_scores >= threshold_value
    ).astype(int)

    threshold_precision = precision_score(
        y,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y,
        threshold_predictions,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y,
        threshold_predictions,
        zero_division=0
    )

    cm_threshold = confusion_matrix(
        y,
        threshold_predictions
    )

    tn_t, fp_t, fn_t, tp_t = (
        cm_threshold.ravel()
    )

    st.subheader(
        f"Current Threshold: {threshold_value:.3f}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Precision",
            f"{threshold_precision * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Recall",
            f"{threshold_recall * 100:.2f}%"
        )

    with col3:

        st.metric(
            "F1 Score",
            f"{threshold_f1 * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Fraud Detected",
            f"{tp_t:,}"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Missed",
            f"{fn_t:,}"
        )

    with col2:

        st.metric(
            "False Positives",
            f"{fp_t:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # THRESHOLD COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "📊 Threshold Comparison"
    )

    comparison_thresholds = [
        0.010,
        0.020,
        0.040,
        0.060,
        0.080,
        0.100,
        saved_threshold
    ]

    threshold_rows = []

    for t in comparison_thresholds:

        pred = (
            all_scores >= t
        ).astype(int)

        p = precision_score(
            y,
            pred,
            zero_division=0
        )

        r = recall_score(
            y,
            pred,
            zero_division=0
        )

        f = f1_score(
            y,
            pred,
            zero_division=0
        )

        threshold_rows.append({
            "Threshold": t,
            "Precision": p,
            "Recall": r,
            "F1 Score": f
        })

    threshold_table = pd.DataFrame(
        threshold_rows
    ).drop_duplicates(
        subset=["Threshold"]
    )

    st.dataframe(
        threshold_table.style.format({
            "Threshold": "{:.6f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # PR CURVE
    # --------------------------------------------------------

    st.subheader(
        "📈 Precision-Recall Curve"
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y,
            all_scores
        )
    )

    average_precision = (
        average_precision_score(
            y,
            all_scores
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"Isolation Forest "
              f"(AP = {average_precision:.4f})"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curve"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.info(
        """
        Higher thresholds generally reduce the number of
        transactions flagged as anomalous, while lower
        thresholds increase sensitivity. The selected project
        threshold is based on the tested threshold analysis.
        """
    )


# ============================================================
# PAGE 5 - MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.header("🤖 Model Comparison Center")

    st.write(
        "Side-by-side comparison of the two unsupervised "
        "anomaly detection approaches used in the project."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    comparison_df = pd.DataFrame({

        "Model": [
            "Isolation Forest",
            "LOF"
        ],

        "Precision": [
            0.137544,
            0.010300
        ],

        "Recall": [
            0.398374,
            0.247967
        ],

        "F1 Score": [
            0.204486,
            0.019778
        ]

    })

    st.dataframe(
        comparison_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    st.subheader(
        "📊 Performance Comparison"
    )

    chart_df = comparison_df.set_index(
        "Model"
    )

    st.bar_chart(
        chart_df[
            [
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌲 Isolation Forest"
        )

        st.markdown(
            """
            **Approach**

            Isolation-based anomaly detection.

            **Training**

            Primarily trained using normal transactions.

            **Project Result**

            - Precision: 13.75%
            - Recall: 39.84%
            - F1: 20.45%

            **Selected Model**

            🏆 Isolation Forest
            """
        )

    with col2:

        st.subheader(
            "📍 Local Outlier Factor"
        )

        st.markdown(
            """
            **Approach**

            Local-density-based anomaly detection.

            **Training**

            Evaluated using the normal transaction population.

            **Project Result**

            - Precision: 1.03%
            - Recall: 24.80%
            - F1: 1.98%

            **Comparison**

            Lower F1 than Isolation Forest in this project.
            """
        )


# ============================================================
# PAGE 6 - DATASET ANALYTICS
# ============================================================

elif page == "📊 Dataset Analytics":

    st.header("📊 Dataset Analytics")

    st.write(
        "Explore the underlying transaction dataset and "
        "fraud distribution."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{df.shape[1]}"
        )

    with col3:

        st.metric(
            "Normal",
            f"{normal_transactions:,}"
        )

    with col4:

        st.metric(
            "Fraud",
            f"{fraud_transactions:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "⚖️ Class Distribution"
    )

    class_data = pd.DataFrame({
        "Class": [
            "Normal",
            "Fraud"
        ],
        "Count": [
            normal_transactions,
            fraud_transactions
        ]
    })

    st.bar_chart(
        class_data.set_index(
            "Class"
        )
    )

    st.divider()

    # --------------------------------------------------------
    # AMOUNT ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "💰 Transaction Amount Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "Normal Transaction Amounts"
        )

        normal_amount = df[
            df["Class"] == 0
        ]["Amount"]

        st.line_chart(
            normal_amount.head(5000)
        )

    with col2:

        st.write(
            "Fraud Transaction Amounts"
        )

        fraud_amount = df[
            df["Class"] == 1
        ]["Amount"]

        st.line_chart(
            fraud_amount.reset_index(
                drop=True
            )
        )

    st.divider()

    # --------------------------------------------------------
    # AMOUNT STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "💵 Amount Statistics"
    )

    amount_stats = df.groupby(
        "Class"
    )["Amount"].describe()

    amount_stats.index = [
        "Normal",
        "Fraud"
    ]

    st.dataframe(
        amount_stats,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # TIME ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "⏱️ Transaction Time Analysis"
    )

    time_data = df.groupby(
        "Class"
    )["Time"].mean()

    time_data.index = [
        "Normal",
        "Fraud"
    ]

    st.bar_chart(
        time_data
    )

    st.divider()

    # --------------------------------------------------------
    # DATA SAMPLE
    # --------------------------------------------------------

    st.subheader(
        "📋 Dataset Sample"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 7 - PCA
# ============================================================

elif page == "🔬 PCA Visualization":

    st.header("🔬 PCA Visualization")

    st.write(
        "Dimensionality reduction view of the transaction "
        "feature space."
    )

    st.divider()

    # --------------------------------------------------------
    # PCA 2D
    # --------------------------------------------------------

    @st.cache_data
    def calculate_pca_2d(_X):

        pca = PCA(
            n_components=2
        )

        transformed = pca.fit_transform(
            _X
        )

        return (
            transformed,
            pca.explained_variance_ratio_
        )


    X_pca, explained_2d = calculate_pca_2d(
        X
    )

    st.subheader(
        "🔵 2D PCA Visualization"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "PC1 Variance",
            f"{explained_2d[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "PC2 Variance",
            f"{explained_2d[1] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Total Explained",
            f"{explained_2d.sum() * 100:.2f}%"
        )

    # Sample for visualization
    sample_size = min(
        15000,
        len(df)
    )

    rng = np.random.default_rng(
        42
    )

    sample_indices = rng.choice(
        len(df),
        size=sample_size,
        replace=False
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    normal_mask = (
        y[sample_indices] == 0
    )

    fraud_mask = (
        y[sample_indices] == 1
    )

    ax.scatter(
        X_pca[sample_indices][normal_mask, 0],
        X_pca[sample_indices][normal_mask, 1],
        s=8,
        alpha=0.35,
        label="Normal"
    )

    ax.scatter(
        X_pca[sample_indices][fraud_mask, 0],
        X_pca[sample_indices][fraud_mask, 1],
        s=18,
        alpha=0.8,
        label="Fraud"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "Transaction Distribution in 2D PCA Space"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        f"Visualization uses a random sample of "
        f"{sample_size:,} transactions."
    )

    st.divider()

    # --------------------------------------------------------
    # PCA 3D
    # --------------------------------------------------------

    st.subheader(
        "🧊 3D PCA Visualization"
    )

    @st.cache_data
    def calculate_pca_3d(_X):

        pca = PCA(
            n_components=3
        )

        transformed = pca.fit_transform(
            _X
        )

        return (
            transformed,
            pca.explained_variance_ratio_
        )


    X_pca_3d, explained_3d = calculate_pca_3d(
        X
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "PC1",
            f"{explained_3d[0] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "PC2",
            f"{explained_3d[1] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "PC3",
            f"{explained_3d[2] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Total",
            f"{explained_3d.sum() * 100:.2f}%"
        )

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        X_pca_3d[sample_indices][normal_mask, 0],
        X_pca_3d[sample_indices][normal_mask, 1],
        X_pca_3d[sample_indices][normal_mask, 2],
        s=5,
        alpha=0.25,
        label="Normal"
    )

    ax.scatter(
        X_pca_3d[sample_indices][fraud_mask, 0],
        X_pca_3d[sample_indices][fraud_mask, 1],
        X_pca_3d[sample_indices][fraud_mask, 2],
        s=18,
        alpha=0.8,
        label="Fraud"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    ax.set_title(
        "Transaction Distribution in 3D PCA Space"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# PAGE 8 - DETECTION HISTORY
# ============================================================

elif page == "📋 Detection History":

    st.header("📋 Detection History")

    st.write(
        "Transactions analyzed during the current application session."
    )

    st.divider()

    if len(st.session_state.history) == 0:

        st.info(
            """
            No transactions have been analyzed yet.

            Go to **Transaction Investigation** and analyze
            a transaction to create detection history.
            """
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Transactions Analyzed",
                len(history_df)
            )

        with col2:

            anomalies = (
                history_df["Prediction"]
                .str.contains("ANOMALY")
                .sum()
            )

            st.metric(
                "Anomalies",
                int(anomalies)
            )

        with col3:

            normal_count = (
                len(history_df) -
                anomalies
            )

            st.metric(
                "Normal",
                int(normal_count)
            )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "📈 Anomaly Scores"
        )

        chart_history = history_df[
            [
                "Transaction ID",
                "Anomaly Score"
            ]
        ].set_index(
            "Transaction ID"
        )

        st.line_chart(
            chart_history
        )

        if st.button(
            "🗑️ Clear Detection History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💳 Financial AI Command Center | "
    "Isolation Forest + LOF | "
    "Real-Time Financial Anomaly Detection"
)