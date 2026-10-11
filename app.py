import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Financial AI Command Center",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CASES_PATH = RESULTS_DIR / "investigation_cases.csv"

IF_PATH = MODELS_DIR / "isolation_forest.pkl"
LOF_PATH = MODELS_DIR / "lof.pkl"
IF_THRESHOLD_PATH = MODELS_DIR / "isolation_forest_threshold.pkl"
AMOUNT_SCALER_PATH = MODELS_DIR / "amount_scaler.pkl"
TIME_SCALER_PATH = MODELS_DIR / "time_scaler.pkl"
AE_PATH = MODELS_DIR / "autoencoder.keras"
AE_THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.pkl"

FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>

/* Fix top heading being clipped */
header[data-testid="stHeader"] {
    height: 3.5rem !important;
    background: transparent !important;
}

.stApp [data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}

h1 {
    line-height: 1.4 !important;
    padding-top: 0.4rem !important;
    overflow: visible !important;
}

:root { --red:#ff2438; --panel:#0b1117; --line:#1c2a34; --muted:#a9b7c4; }
.stApp { background:radial-gradient(ellipse at 75% 0%,#19070c 0%,#070b10 42%,#030609 100%); color:#f4f7fa; }
.block-container {padding-top:1.1rem;padding-bottom:2rem;max-width:1600px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#080d12 0%,#030608 100%);border-right:1px solid #1c2831;}
[data-testid="stSidebar"] * {color:#eef3f7;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {border-radius:7px;padding:5px 7px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#241016;}
.main-title {font-size:31px;font-weight:850;letter-spacing:-.7px;margin-bottom:1px;}
.main-title span {color:var(--red);}
.subtitle {color:#aab6c2;font-size:12px;letter-spacing:.45px;margin-bottom:18px;}
.hero-banner {padding:22px 25px;border:1px solid #6d1420;border-radius:15px;background:linear-gradient(105deg,#21070d 0%,#100b10 48%,#300710 100%);box-shadow:0 0 28px #ff24380d inset;margin:8px 0 18px;}
.hero-title {font-size:24px;font-weight:800;line-height:1.2;}
.hero-title span {color:#ff3548;}
.hero-sub {color:#c4ccd5;font-size:13px;margin-top:7px;}
.kpi-card {background:linear-gradient(145deg,#0e151c,#080d12);border:1px solid #1b2a35;border-radius:13px;padding:17px;min-height:126px;}
.kpi-label {font-size:11px;color:#b9c5cf;margin-bottom:10px;letter-spacing:.5px;}
.kpi-value {font-size:26px;font-weight:800;letter-spacing:-.5px;color:#f6f8fa;}
.kpi-foot {font-size:11px;color:#a2b0bd;margin-top:7px;}
.kpi-icon {float:right;font-size:21px;color:#ff3347;background:#2b0b12;border-radius:9px;padding:7px 9px;}
.panel {background:linear-gradient(145deg,#0c1319,#080c11);border:1px solid #1b2a35;border-radius:13px;padding:16px 18px;margin-bottom:12px;}
.panel-title {font-size:16px;font-weight:750;margin-bottom:10px;}
.red-pill {display:inline-block;background:#4a0912;color:#ff6a78;border:1px solid #7d1725;border-radius:5px;padding:4px 8px;font-size:10px;font-weight:800;letter-spacing:.5px;}
.risk-high {padding:16px;border-radius:12px;background:linear-gradient(120deg,#26080f,#100b10);border:1px solid #ff2438;}
.risk-normal {padding:16px;border-radius:12px;background:#071b14;border:1px solid #1b7753;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#0d141b,#080c11);border:1px solid #1b2a35;border-radius:12px;padding:13px;}
div[data-testid="stMetricLabel"] {color:#b9c5cf;}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#ff2438,#c90020);border:0;color:white;border-radius:8px;font-weight:700;}
.stButton > button {border-radius:8px;}
hr {border-color:#1c2a34;}
h1,h2,h3 {letter-spacing:-.3px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOADERS
# ============================================================
@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file missing: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)

@st.cache_resource
def load_models():
    required = [IF_PATH, LOF_PATH, IF_THRESHOLD_PATH, AMOUNT_SCALER_PATH, TIME_SCALER_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required model files: " + ", ".join(missing))

    isolation_forest = joblib.load(IF_PATH)
    lof = joblib.load(LOF_PATH)
    if_threshold = float(np.asarray(joblib.load(IF_THRESHOLD_PATH)).reshape(-1)[0])
    amount_scaler = joblib.load(AMOUNT_SCALER_PATH)
    time_scaler = joblib.load(TIME_SCALER_PATH)

    autoencoder = None
    ae_threshold = None
    if AE_PATH.exists() and AE_THRESHOLD_PATH.exists():
        autoencoder = load_model(AE_PATH, compile=False)
        ae_threshold = float(np.asarray(joblib.load(AE_THRESHOLD_PATH)).reshape(-1)[0])

    return isolation_forest, lof, if_threshold, amount_scaler, time_scaler, autoencoder, ae_threshold

def prepare_features(rows, amount_scaler, time_scaler):
    """Scale raw dataset rows in the same way as the notebooks."""
    frame = rows[FEATURE_ORDER].copy()
    frame["Amount"] = amount_scaler.transform(frame[["Amount"]])
    frame["Time"] = time_scaler.transform(frame[["Time"]])
    return frame[FEATURE_ORDER].astype("float32")

def ae_errors(model, features):
    reconstructed = model.predict(features, verbose=0)
    return np.mean(np.square(features.to_numpy() - reconstructed), axis=1)

def score_models(features, isolation_forest, lof, if_threshold, autoencoder, ae_threshold):
    if_scores = -isolation_forest.decision_function(features)
    if_flags = if_scores >= if_threshold
    lof_scores = -lof.decision_function(features)
    lof_flags = lof.predict(features) == -1
    ae_scores = ae_flags = None
    if autoencoder is not None:
        ae_scores = ae_errors(autoencoder, features)
        ae_flags = ae_scores >= ae_threshold
    return if_scores, if_flags, lof_scores, lof_flags, ae_scores, ae_flags

def metric_table(y_true, predictions):
    return {
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 score": f1_score(y_true, predictions, zero_division=0),
    }

# ============================================================
# INITIALIZE APP
# ============================================================
st.markdown('<div class="main-title">🛡️ Financial <span>AI</span> Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SMARTER DETECTION&nbsp;&nbsp; | &nbsp;&nbsp; SAFER TRANSACTIONS&nbsp;&nbsp; | &nbsp;&nbsp; A SECURE FINANCIAL FUTURE</div>', unsafe_allow_html=True)

try:
    df = load_dataset()
    (isolation_forest, lof, if_threshold, amount_scaler, time_scaler,
     autoencoder, ae_threshold) = load_models()
except Exception as exc:
    st.error(f"Could not initialize dashboard: {exc}")
    st.info("Confirm dataset/creditcard.csv and the trained files inside models/ are present.")
    st.stop()

with st.sidebar:
    st.markdown("## 💳 Financial AI")
    st.caption("Anomaly Detection Command Center")
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
            "📋 Detection History",
            "🧠 AI Investigation Assistant",
            "🛡️ Case Management",
            "⚡ Live-Style Monitor",
            "📄 Investigation Report",
        ],
    )
    st.divider()
    st.caption("Loaded models")
    st.write("✅ Isolation Forest")
    st.write("✅ Local Outlier Factor")
    st.write("✅ Autoencoder" if autoencoder is not None else "⚠️ Autoencoder files unavailable")

# ============================================================
# SHARED DATA
# ============================================================
y = df["Class"].astype(int)
total_count = len(df)
fraud_count = int(y.sum())
normal_count = total_count - fraud_count

# ============================================================
# 1. OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Welcome to <span>Financial AI Command Center</span></div>
      <div class="hero-sub">Advanced anomaly detection and investigation for safer financial transactions.</div>
    </div>
    """, unsafe_allow_html=True)

    fraud_rate = (fraud_count / total_count * 100) if total_count else 0
    normal_rate = (normal_count / total_count * 100) if total_count else 0
    k1, k2, k3, k4 = st.columns(4)
    cards = [
        (k1, "TOTAL TRANSACTIONS", f"{total_count:,}", "Dataset records loaded", "▤"),
        (k2, "FRAUD-LABELLED", f"{fraud_count:,}", f"{fraud_rate:.3f}% of dataset", "⚠"),
        (k3, "NORMAL TRANSACTIONS", f"{normal_count:,}", f"{normal_rate:.2f}% of dataset", "⬡"),
        (k4, "MODELS AVAILABLE", str(2 + int(autoencoder is not None)), "Isolation Forest · LOF · Autoencoder", "◉"),
    ]
    for col, label, value, foot, icon in cards:
        with col:
            st.markdown(f'<div class="kpi-card"><span class="kpi-icon">{icon}</span><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, mid, right = st.columns([1.45, 1, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Transaction Class Distribution</div>', unsafe_allow_html=True)
        class_counts = pd.DataFrame({"Transactions": [normal_count, fraud_count]}, index=["Normal", "Fraud-labelled"])
        st.bar_chart(class_counts, height=230, color="#ff2438")
        st.markdown('</div>', unsafe_allow_html=True)
    with mid:
        st.markdown('<div class="panel"><div class="panel-title">Anomaly Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:34px;font-weight:800;color:#ff3347;">{fraud_count:,}</div><div style="color:#b7c2cc;font-size:12px;">Fraud-labelled dataset rows</div><br><div class="red-pill">RARE-CLASS DETECTION</div><br><br><div style="font-size:12px;color:#b7c2cc;">Fraud rate</div><div style="font-size:22px;font-weight:750;">{fraud_rate:.3f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Model Status</div>', unsafe_allow_html=True)
        st.markdown(f'<p>🟢 <b>Isolation Forest</b><br><span style="color:#8fa0ad;font-size:12px;">Threshold: {if_threshold:.5f}</span></p><p>🟢 <b>Local Outlier Factor</b></p><p>{"🟢" if autoencoder is not None else "🟠"} <b>Autoencoder</b><br><span style="color:#8fa0ad;font-size:12px;">{"Threshold: " + format(ae_threshold, ".5f") if ae_threshold is not None else "Model files unavailable"}</span></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    lower1, lower2 = st.columns([1.1, 1])
    with lower1:
        st.markdown('<div class="panel"><div class="panel-title">Model Performance Snapshot · F1-score</div>', unsafe_allow_html=True)
        comparison_path = RESULTS_DIR / "final_model_comparison.csv"
        if comparison_path.exists():
            try:
                comparison_data = pd.read_csv(comparison_path)
                lower_cols = {c.lower(): c for c in comparison_data.columns}
                model_col = next((lower_cols[k] for k in lower_cols if "model" in k), None)
                f1_col = next((lower_cols[k] for k in lower_cols if "f1" in k), None)
                if model_col and f1_col:
                    chart_df = comparison_data[[model_col, f1_col]].dropna().copy().set_index(model_col)
                    st.bar_chart(chart_df, height=210, color="#ff2438")
                else:
                    st.info("Comparison CSV found, but expected Model and F1 columns were not identified.")
            except Exception:
                st.info("Open Model Comparison to view the recorded metrics.")
        else:
            st.info("Model comparison file not found. Open Model Comparison for the current reference metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
    with lower2:
        st.markdown('<div class="panel"><div class="panel-title">Investigation Activity</div><div style="color:#b6c2cc;font-size:13px;line-height:2.0;">🔴 &nbsp; Flag suspicious transactions for review<br>🧠 &nbsp; Compare three anomaly models<br>🛡️ &nbsp; Track investigation cases and notes<br>📄 &nbsp; Export investigation summaries</div></div>', unsafe_allow_html=True)

    st.warning("This dashboard supports investigation; an anomaly flag does not prove fraud. Evaluate final model performance on a separate held-out test set before production use.")
    st.caption("Dataset: creditcard.csv. V1–V28 are PCA-transformed features and are not direct business variables.")

# ============================================================
# 2. TRANSACTION INVESTIGATION
# ============================================================
elif page == "🔍 Transaction Investigation":
    st.subheader("🔍 Transaction Investigation")
    st.write("Select a dataset row and compare the model outputs.")
    index = st.number_input("Transaction row index", min_value=0, max_value=len(df)-1, value=0, step=1)
    raw = df.iloc[int(index)]
    features = prepare_features(df.iloc[[int(index)]], amount_scaler, time_scaler)

    st.write(f"**Amount:** ${float(raw['Amount']):,.2f}  |  **Dataset label:** {'Fraud' if int(raw['Class']) else 'Normal'}")
    if st.button("Analyze selected transaction", type="primary"):
        scores = score_models(features, isolation_forest, lof, if_threshold, autoencoder, ae_threshold)
        if_s, if_f, lof_s, lof_f, ae_s, ae_f = scores
        cols = st.columns(3)
        with cols[0]:
            st.markdown("#### Isolation Forest")
            st.metric("Anomaly score", f"{if_s[0]:.5f}")
            st.warning("Flagged anomaly" if if_f[0] else "Predicted normal")
        with cols[1]:
            st.markdown("#### LOF")
            st.metric("Anomaly score", f"{lof_s[0]:.5f}")
            st.warning("Flagged anomaly" if lof_f[0] else "Predicted normal")
        with cols[2]:
            st.markdown("#### Autoencoder")
            if ae_s is None:
                st.warning("Model files unavailable")
            else:
                st.metric("Reconstruction error", f"{ae_s[0]:.5f}")
                st.warning("Flagged anomaly" if ae_f[0] else "Predicted normal")
                st.caption(f"Threshold: {ae_threshold:.5f}")
        if autoencoder is not None:
            reconstructed = autoencoder.predict(features, verbose=0)[0]
            errors = np.square(features.iloc[0].to_numpy() - reconstructed)
            error_df = pd.DataFrame({"Feature": FEATURE_ORDER, "Squared reconstruction error": errors})
            error_df = error_df.sort_values("Squared reconstruction error", ascending=False).head(10)
            st.markdown("### Largest reconstruction deviations")
            st.bar_chart(error_df.set_index("Feature"))
            st.caption("These deviations are not causal explanations or probabilities of fraud.")

# ============================================================
# 3. FRAUD DETECTION
# ============================================================
elif page == "🚨 Fraud Detection":
    st.subheader("🚨 Fraud Detection")
    st.write("Run the selected model over a sample of transactions. This may take a little time.")
    sample_size = st.slider("Number of transactions to scan", min_value=500, max_value=min(20000, len(df)), value=min(3000, len(df)), step=500)
    model_choice = st.selectbox("Detection model", ["Autoencoder", "Isolation Forest", "Local Outlier Factor"])
    if model_choice == "Autoencoder" and autoencoder is None:
        st.warning("Autoencoder files are unavailable. Choose another model.")
    elif st.button("Run fraud scan", type="primary"):
        sample = df.sample(n=int(sample_size), random_state=42).copy()
        Xs = prepare_features(sample, amount_scaler, time_scaler)
        if model_choice == "Autoencoder":
            sample["Anomaly Score"] = ae_errors(autoencoder, Xs)
            sample["Predicted Anomaly"] = sample["Anomaly Score"] >= ae_threshold
        elif model_choice == "Isolation Forest":
            sample["Anomaly Score"] = -isolation_forest.decision_function(Xs)
            sample["Predicted Anomaly"] = sample["Anomaly Score"] >= if_threshold
        else:
            sample["Anomaly Score"] = -lof.decision_function(Xs)
            sample["Predicted Anomaly"] = lof.predict(Xs) == -1
        sample["Actual Label"] = sample["Class"].map({0: "Normal", 1: "Fraud"})
        st.session_state["last_detection_results"] = sample
        st.session_state["last_detection_model"] = model_choice
        flagged = int(sample["Predicted Anomaly"].sum())
        st.metric("Flagged transactions", f"{flagged:,}", f"{flagged / len(sample) * 100:.2f}% of scanned sample")
        st.dataframe(sample.loc[sample["Predicted Anomaly"]].sort_values("Anomaly Score", ascending=False).head(100),
                     use_container_width=True, hide_index=True)
        st.download_button("Download scan results CSV", sample.to_csv(index=False).encode("utf-8"),
                           file_name="fraud_detection_scan.csv", mime="text/csv")

# ============================================================
# 4. THRESHOLD ANALYSIS
# ============================================================
elif page == "🎚️ Threshold Analysis":
    st.subheader("🎚️ Threshold Analysis")
    st.write("Explore the effect of changing the Autoencoder threshold on a labelled sample.")
    if autoencoder is None:
        st.warning("Autoencoder files are unavailable.")
    else:
        sample_size = st.slider("Evaluation sample size", 1000, min(20000, len(df)), min(5000, len(df)), 1000)
        sample = df.sample(n=int(sample_size), random_state=17)
        Xs = prepare_features(sample, amount_scaler, time_scaler)
        errors = ae_errors(autoencoder, Xs)
        threshold = st.slider("Reconstruction-error threshold", float(max(0.000001, errors.min())),
                              float(max(errors.max(), ae_threshold * 1.01)),
                              float(min(ae_threshold, max(errors.max(), ae_threshold * 1.01))),
                              step=float(max(errors.max(), ae_threshold) / 500))
        predictions = (errors >= threshold).astype(int)
        truth = sample["Class"].to_numpy().astype(int)
        p = precision_score(truth, predictions, zero_division=0)
        r = recall_score(truth, predictions, zero_division=0)
        f = f1_score(truth, predictions, zero_division=0)
        a, b, c = st.columns(3)
        a.metric("Precision", f"{p:.4f}")
        b.metric("Recall", f"{r:.4f}")
        c.metric("F1 score", f"{f:.4f}")
        st.write("Confusion matrix (rows = actual [normal, fraud], columns = predicted [normal, anomaly])")
        st.dataframe(pd.DataFrame(confusion_matrix(truth, predictions, labels=[0, 1]),
                                  index=["Actual normal", "Actual fraud"],
                                  columns=["Predicted normal", "Predicted anomaly"]))
        curve_p, curve_r, _ = precision_recall_curve(truth, errors)
        fig, ax = plt.subplots()
        ax.plot(curve_r, curve_p)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall curve for reconstruction error")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 5. MODEL COMPARISON
# ============================================================
elif page == "🤖 Model Comparison":
    st.subheader("🤖 Model Comparison")
    st.caption("Reference metrics from the current project evaluation; use a held-out test set for a fair final report.")
    comparison = pd.DataFrame([
        {"Model": "Autoencoder", "Precision": 0.212942, "Recall": 0.782520, "F1 score": 0.334783},
        {"Model": "Isolation Forest", "Precision": 0.137544, "Recall": 0.398374, "F1 score": 0.204486},
        {"Model": "LOF", "Precision": 0.010300, "Recall": 0.247967, "F1 score": 0.019778},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.markdown("### F1 score comparison")
    st.bar_chart(comparison.set_index("Model")[["F1 score"]])
    st.markdown("### Precision and recall")
    st.bar_chart(comparison.set_index("Model")[["Precision", "Recall"]])
    st.warning("A higher recall may produce more false positives. Compare all metrics, not just one.")

# ============================================================
# 6. DATASET ANALYTICS
# ============================================================
elif page == "📊 Dataset Analytics":
    st.subheader("📊 Dataset Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Missing values")
        missing = df.isna().sum()
        st.dataframe(missing[missing > 0].rename("Missing count").to_frame(), use_container_width=True)
        if not missing.any():
            st.success("No missing values found.")
    with c2:
        st.markdown("### Descriptive statistics")
        st.dataframe(df[["Time", "Amount", "Class"]].describe(), use_container_width=True)
    st.markdown("### Amount distribution by class")
    fig, ax = plt.subplots()
    ax.hist(df.loc[df["Class"] == 0, "Amount"], bins=50, alpha=0.65, label="Normal")
    ax.hist(df.loc[df["Class"] == 1, "Amount"], bins=50, alpha=0.65, label="Fraud")
    ax.set_xlabel("Transaction amount")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("### Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

# ============================================================
# 7. PCA VISUALIZATION
# ============================================================
elif page == "🔬 PCA Visualization":
    st.subheader("🔬 PCA Visualization")
    st.write("Visualize a sample in two dimensions. PCA is for visualization here, not the trained detector.")
    sample_size = st.slider("Rows to visualize", 500, min(10000, len(df)), min(2500, len(df)), 500)
    sample = df.sample(n=int(sample_size), random_state=21)
    Xs = prepare_features(sample, amount_scaler, time_scaler)
    with st.spinner("Computing PCA..."):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                            "Class": sample["Class"].map({0: "Normal", 1: "Fraud"}).to_numpy()})
    fig, ax = plt.subplots()
    for label in ["Normal", "Fraud"]:
        subset = plot_df[plot_df["Class"] == label]
        ax.scatter(subset["PC1"], subset["PC2"], s=8 if label == "Normal" else 18,
                   alpha=0.35 if label == "Normal" else 0.8, label=label)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"Variance explained by 2D PCA: **{pca.explained_variance_ratio_.sum() * 100:.2f}%**")

# ============================================================
# 8. DETECTION HISTORY
# ============================================================
elif page == "📋 Detection History":
    st.subheader("📋 Detection History")
    history = st.session_state.get("last_detection_results")
    history_model = st.session_state.get("last_detection_model")
    if history is None:
        st.info("No scan history in this browser session yet. Go to 🚨 Fraud Detection and run a scan.")
    else:
        st.write(f"Latest scan model: **{history_model}**")
        st.metric("Transactions scanned", f"{len(history):,}")
        st.metric("Flagged anomalies", f"{int(history['Predicted Anomaly'].sum()):,}")
        st.dataframe(history.sort_values("Anomaly Score", ascending=False).head(200),
                     use_container_width=True, hide_index=True)
        st.download_button("Download detection history CSV", history.to_csv(index=False).encode("utf-8"),
                           file_name="detection_history.csv", mime="text/csv")


# ============================================================
# 9. AI INVESTIGATION ASSISTANT (evidence-based, not generative)
# ============================================================
elif page == "🧠 AI Investigation Assistant":
    st.subheader("🧠 AI Investigation Assistant")
    st.caption("Evidence-based investigation support from the selected model; this does not prove fraud.")
    row_id = st.number_input("Transaction row index", min_value=0, max_value=len(df)-1, value=0, step=1, key="assistant_row")
    raw = df.iloc[[int(row_id)]]
    features = prepare_features(raw, amount_scaler, time_scaler)
    selected_model = st.selectbox("Model for investigation", ["Autoencoder", "Isolation Forest", "Local Outlier Factor"], key="assistant_model")
    if selected_model == "Autoencoder" and autoencoder is None:
        st.warning("Autoencoder files are unavailable. Select another model.")
    elif st.button("Generate investigation insights", type="primary"):
        if selected_model == "Autoencoder":
            score = float(ae_errors(autoencoder, features)[0]); threshold = float(ae_threshold); flagged = score >= threshold
            reconstructed = autoencoder.predict(features, verbose=0)[0]
            deviations = np.square(features.iloc[0].to_numpy() - reconstructed)
            explanation = "reconstruction error"
        elif selected_model == "Isolation Forest":
            score = float(-isolation_forest.decision_function(features)[0]); threshold = float(if_threshold); flagged = score >= threshold
            explanation = "Isolation Forest anomaly score"
            deviations = None
        else:
            score = float(-lof.decision_function(features)[0]); flagged = bool(lof.predict(features)[0] == -1)
            threshold = None; explanation = "LOF outlier prediction"
            deviations = None
        a, b, c = st.columns(3)
        a.metric("Model", selected_model)
        b.metric("Score", f"{score:.6f}")
        c.metric("Dataset label (for evaluation)", "Fraud" if int(raw.iloc[0]["Class"]) else "Normal")
        if flagged:
            st.error("🚨 Model flagged this transaction for manual review.")
        else:
            st.success("No anomaly flag from this model. This does not guarantee the transaction is safe.")
        if threshold is not None:
            st.write(f"Configured threshold: **{threshold:.6f}**")
        st.markdown("### Investigation summary")
        st.write(f"The selected model's {explanation} is **{score:.6f}**. " +
                 ("It crosses the configured threshold and should be reviewed." if flagged else "It does not cross the configured threshold."))
        st.markdown("### Suggested review checklist")
        st.markdown("- Verify transaction and account details using authorized systems.\n- Check recent activity and known customer context.\n- Review related transactions before making a decision.\n- Record the investigator's findings and final decision.")
        if deviations is not None:
            error_df = pd.DataFrame({"Feature": FEATURE_ORDER, "Squared reconstruction error": deviations}).sort_values("Squared reconstruction error", ascending=False).head(10)
            st.markdown("### Largest feature-level reconstruction deviations")
            st.bar_chart(error_df.set_index("Feature"))
            st.caption("V1–V28 are PCA-transformed variables. Deviations are model evidence, not causal explanations.")
        st.session_state["assistant_last_report"] = {
            "row_id": int(row_id), "amount": float(raw.iloc[0]["Amount"]), "model": selected_model,
            "score": score, "threshold": threshold, "flagged": bool(flagged),
            "actual_label": int(raw.iloc[0]["Class"])
        }

# ============================================================
# 10. CASE MANAGEMENT
# ============================================================
elif page == "🛡️ Case Management":
    st.subheader("🛡️ Investigation Case Management")
    st.write("Create and track review cases for flagged transactions. Cases are stored locally in results/investigation_cases.csv.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if CASES_PATH.exists():
        try:
            cases = pd.read_csv(CASES_PATH)
        except Exception:
            cases = pd.DataFrame(columns=["Case ID", "Transaction Row", "Model", "Risk Score", "Status", "Notes", "Created At"])
    else:
        cases = pd.DataFrame(columns=["Case ID", "Transaction Row", "Model", "Risk Score", "Status", "Notes", "Created At"])
    with st.form("new_case_form"):
        case_row = st.number_input("Transaction row to investigate", min_value=0, max_value=len(df)-1, value=0, step=1, key="case_row")
        case_model = st.selectbox("Model", ["Autoencoder", "Isolation Forest", "Local Outlier Factor"], key="case_model")
        case_status = st.selectbox("Initial status", ["Open", "Under Review", "Escalated", "Resolved"])
        case_notes = st.text_area("Investigator notes", placeholder="Record what needs to be verified...")
        create_case = st.form_submit_button("Create investigation case", type="primary")
    if create_case:
        row_features = prepare_features(df.iloc[[int(case_row)]], amount_scaler, time_scaler)
        if case_model == "Autoencoder":
            if autoencoder is None:
                st.error("Autoencoder is unavailable. Choose another model.")
                st.stop()
            case_score = float(ae_errors(autoencoder, row_features)[0])
        elif case_model == "Isolation Forest":
            case_score = float(-isolation_forest.decision_function(row_features)[0])
        else:
            case_score = float(-lof.decision_function(row_features)[0])
        new_id = f"FRD-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
        new_case = {"Case ID": new_id, "Transaction Row": int(case_row), "Model": case_model,
                    "Risk Score": case_score, "Status": case_status, "Notes": case_notes,
                    "Created At": pd.Timestamp.now().isoformat(timespec="seconds")}
        cases = pd.concat([cases, pd.DataFrame([new_case])], ignore_index=True)
        cases.to_csv(CASES_PATH, index=False)
        st.success(f"Case {new_id} created and saved.")
    if not cases.empty:
        st.markdown("### Existing cases")
        st.dataframe(cases, use_container_width=True, hide_index=True)
        case_ids = cases["Case ID"].astype(str).tolist()
        selected_case_id = st.selectbox("Select a case to update", case_ids)
        selected_case = cases[cases["Case ID"].astype(str) == selected_case_id].iloc[0]
        with st.form("update_case_form"):
            updated_status = st.selectbox("Update status", ["Open", "Under Review", "Escalated", "Resolved"],
                                          index=["Open", "Under Review", "Escalated", "Resolved"].index(selected_case["Status"])
                                          if selected_case["Status"] in ["Open", "Under Review", "Escalated", "Resolved"] else 0)
            updated_notes = st.text_area("Update notes", value=str(selected_case.get("Notes", "")))
            update_case = st.form_submit_button("Save case update")
        if update_case:
            mask = cases["Case ID"].astype(str) == selected_case_id
            cases.loc[mask, "Status"] = updated_status
            cases.loc[mask, "Notes"] = updated_notes
            cases.to_csv(CASES_PATH, index=False)
            st.success("Case updated.")
            st.rerun()
        st.download_button("Download cases CSV", cases.to_csv(index=False).encode("utf-8"),
                           file_name="investigation_cases.csv", mime="text/csv")
    else:
        st.info("No cases created yet. Use the form above to create your first case.")

# ============================================================
# 11. LIVE-STYLE MONITOR (simulated replay, not a live bank feed)
# ============================================================
elif page == "⚡ Live-Style Monitor":
    st.subheader("⚡ Transaction Monitoring — Simulated Replay")
    st.info("This demo replays rows from the local dataset. It is not connected to a bank or a live transaction stream.")
    replay_count = st.slider("Transactions to replay", 100, min(5000, len(df)), min(1000, len(df)), 100)
    replay_model = st.selectbox("Monitoring model", ["Autoencoder", "Isolation Forest", "Local Outlier Factor"], key="replay_model")
    if replay_model == "Autoencoder" and autoencoder is None:
        st.warning("Autoencoder files are unavailable. Choose another model.")
    elif st.button("▶ Replay transaction batch", type="primary"):
        replay = df.sample(n=int(replay_count), random_state=None).copy()
        replay_x = prepare_features(replay, amount_scaler, time_scaler)
        if replay_model == "Autoencoder":
            replay["Anomaly Score"] = ae_errors(autoencoder, replay_x)
            replay["Predicted Anomaly"] = replay["Anomaly Score"] >= ae_threshold
        elif replay_model == "Isolation Forest":
            replay["Anomaly Score"] = -isolation_forest.decision_function(replay_x)
            replay["Predicted Anomaly"] = replay["Anomaly Score"] >= if_threshold
        else:
            replay["Anomaly Score"] = -lof.decision_function(replay_x)
            replay["Predicted Anomaly"] = lof.predict(replay_x) == -1
        replay["Transaction Row"] = replay.index
        replay = replay.reset_index(drop=True)
        st.session_state["monitor_results"] = replay
        st.session_state["monitor_model"] = replay_model
    monitor = st.session_state.get("monitor_results")
    if monitor is not None:
        flagged = monitor[monitor["Predicted Anomaly"]]
        a, b, c = st.columns(3)
        a.metric("Rows replayed", f"{len(monitor):,}")
        b.metric("Flagged for review", f"{len(flagged):,}")
        c.metric("Flag rate", f"{len(flagged) / max(len(monitor), 1) * 100:.2f}%")
        st.markdown("### Highest-scoring flagged transactions")
        st.dataframe(flagged.sort_values("Anomaly Score", ascending=False).head(50), use_container_width=True, hide_index=True)
        st.download_button("Download replay results CSV", monitor.to_csv(index=False).encode("utf-8"),
                           file_name="simulated_monitoring_results.csv", mime="text/csv")

# ============================================================
# 12. INVESTIGATION REPORT
# ============================================================
elif page == "📄 Investigation Report":
    st.subheader("📄 Investigation Report Generator")
    st.write("Generate a downloadable text report for a selected transaction. The report documents model evidence; it does not declare fraud as proven.")
    report_row = st.number_input("Transaction row index", min_value=0, max_value=len(df)-1, value=0, step=1, key="report_row")
    report_model = st.selectbox("Model for report", ["Autoencoder", "Isolation Forest", "Local Outlier Factor"], key="report_model")
    if report_model == "Autoencoder" and autoencoder is None:
        st.warning("Autoencoder files are unavailable. Choose another model.")
    elif st.button("Prepare report", type="primary"):
        raw = df.iloc[[int(report_row)]]
        features = prepare_features(raw, amount_scaler, time_scaler)
        if report_model == "Autoencoder":
            score = float(ae_errors(autoencoder, features)[0]); threshold = float(ae_threshold); flagged = score >= threshold
        elif report_model == "Isolation Forest":
            score = float(-isolation_forest.decision_function(features)[0]); threshold = float(if_threshold); flagged = score >= threshold
        else:
            score = float(-lof.decision_function(features)[0]); threshold = None; flagged = bool(lof.predict(features)[0] == -1)
        report_text = f"FINANCIAL ANOMALY INVESTIGATION REPORT\nGenerated: {pd.Timestamp.now().isoformat(timespec='seconds')}\nTransaction row: {int(report_row)}\nAmount (dataset units): {float(raw.iloc[0]['Amount']):.2f}\nDataset label (evaluation only): {'Fraud' if int(raw.iloc[0]['Class']) else 'Normal'}\nModel: {report_model}\nAnomaly score: {score:.8f}\nThreshold: {threshold if threshold is not None else 'Model-native outlier decision'}\nModel flagged for review: {'YES' if flagged else 'NO'}\n\nSuggested next steps:\n1. Verify transaction and account details in authorized systems.\n2. Review related transactions and contextual information.\n3. Record investigator findings before taking action.\n\nLimitations: An anomaly flag is not proof of fraud. This educational prototype is not connected to live banking systems. V1-V28 are PCA-transformed features.\n"
        st.text_area("Report preview", report_text, height=300)
        st.download_button("Download investigation report (.txt)", report_text.encode("utf-8"),
                           file_name=f"investigation_report_row_{int(report_row)}.txt", mime="text/plain")


st.divider()
st.caption("Educational prototype. An anomaly flag is not proof of fraud and this app is not approved for live financial decisions.")
