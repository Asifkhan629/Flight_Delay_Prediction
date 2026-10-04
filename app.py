"""
app.py
------
Flight Delay Prediction System - Streamlit Web Application

A single-file, professional dashboard that:
  - Loads the trained model (model.pkl) produced by train_model.py
  - Lets users enter flight details and predicts delay probability
  - Displays dataset insights, model performance, and charts
  - Keeps a session-based prediction history with CSV download

Run with:
    streamlit run app.py
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR PALETTE / GLOBAL STYLE (CSS injected directly here)
# ============================================================
BG_COLOR = "#0F172A"
CARD_COLOR = "#1E293B"
PRIMARY = "#3B82F6"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
TEXT_COLOR = "#FFFFFF"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD_COLOR};
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {TEXT_COLOR};
    }}
    .metric-card {{
        background-color: {CARD_COLOR};
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        border: 1px solid #2C3B52;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 800;
    }}
    .metric-label {{
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }}
    .result-card {{
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .section-card {{
        background-color: {CARD_COLOR};
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        border: 1px solid #2C3B52;
    }}
    .stButton>button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        background-color: #2563EB;
        color: white;
    }}
    footer {{visibility: hidden;}}
    .custom-footer {{
        text-align: center;
        color: #64748B;
        padding: 18px 0 6px 0;
        font-size: 13px;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
DATA_PATH = "flights.csv"
MODEL_PATH = "model.pkl"

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm", "Snow"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]


# ============================================================
# DATA / MODEL LOADING (cached for performance)
# ============================================================
@st.cache_data
def load_dataset():
    """Loads the flights dataset from disk."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_bundle():
    """Loads the trained model bundle (model + encoders + metrics)."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def safe_encode(encoder, value):
    """
    Encodes a categorical value using a fitted LabelEncoder.
    Falls back to the first known class if an unseen value is given.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([encoder.classes_[0]])[0]


# ============================================================
# METRIC CARD HELPER
# ============================================================
def metric_card(label, value, color=PRIMARY):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA + MODEL
# ============================================================
df = load_dataset()
bundle = load_model_bundle()

if df is None or bundle is None:
    st.error(
        "Model or dataset not found. Please run `python train_model.py` "
        "first to generate the dataset and train the model."
    )
    st.stop()

model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
accuracy = bundle["accuracy"]
cm = bundle["confusion_matrix"]
report_dict = bundle["classification_report"]

# Session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ✈️ Flight Delay Predictor")
    st.markdown("Navigate through the dashboard sections below.")
    page = st.radio(
        "Go to",
        ["🏠 Predict Delay", "📊 Dataset Insights", "🧠 Model Performance", "📁 Prediction History"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses a **Random Forest Classifier** trained on historical "
        "flight data to predict whether a flight will be delayed."
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your own flights CSV (optional)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Custom dataset loaded for this session!")

    st.markdown("---")
    with open(MODEL_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Download Trained Model",
            data=f,
            file_name="model.pkl",
            mime="application/octet-stream",
        )

# ============================================================
# PAGE 1: PREDICT DELAY
# ============================================================
if page == "🏠 Predict Delay":
    st.markdown("# ✈️ Flight Delay Prediction System")
    st.markdown(
        "Enter your flight details below to predict the likelihood of delay, "
        "powered by a machine learning model trained on historical flight data."
    )

    # ---- Top metric cards ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Model Accuracy", f"{accuracy * 100:.1f}%", SUCCESS)
    with col2:
        metric_card("Total Flights Analyzed", f"{len(df):,}", PRIMARY)
    with col3:
        delay_rate = df["Delay"].mean() * 100
        metric_card("Historical Delay Rate", f"{delay_rate:.1f}%", WARNING)
    with col4:
        metric_card("Airlines Covered", df["Airline"].nunique(), PRIMARY)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Prediction Form ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Flight Details")

    with st.form("prediction_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            airline = st.selectbox("Airline", sorted(df["Airline"].unique()))
            source = st.selectbox("Source City", sorted(df["Source"].unique()))
            destination = st.selectbox(
                "Destination City",
                [c for c in sorted(df["Destination"].unique()) if c != source] or sorted(df["Destination"].unique())
            )
        with f2:
            distance = st.slider("Distance (miles)", 150, 3000, 800, step=50)
            dep_hour = st.slider("Scheduled Departure (24h)", 0, 23, 9)
            arr_hour = st.slider("Scheduled Arrival (24h)", 0, 23, 12)
        with f3:
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(1, 8)),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                        "Friday", "Saturday", "Sunday"][x - 1],
            )
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda x: ["January", "February", "March", "April", "May", "June",
                                        "July", "August", "September", "October", "November",
                                        "December"][x - 1],
            )
            weather = st.selectbox("Weather Condition", WEATHER_OPTIONS)
            traffic = st.selectbox("Airport Traffic Level", TRAFFIC_LEVELS)

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            submitted = st.form_submit_button("🔮 Predict Delay")
        with btn_col2:
            reset = st.form_submit_button("♻️ Reset Form")

    st.markdown("</div>", unsafe_allow_html=True)

    if reset:
        st.rerun()

    # ---- Random Sample Prediction ----
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🎲 Try a Random Flight Sample")
    random_col1, random_col2 = st.columns([1, 3])
    with random_col1:
        random_clicked = st.button("Generate Random Flight")
    st.markdown("</div>", unsafe_allow_html=True)

    if random_clicked:
        sample = df.sample(1).iloc[0]
        airline, source, destination = sample["Airline"], sample["Source"], sample["Destination"]
        distance, dep_hour, arr_hour = sample["Distance"], sample["Scheduled_Departure"], sample["Scheduled_Arrival"]
        day_of_week, month = sample["Day_of_Week"], sample["Month"]
        weather, traffic = sample["Weather"], sample["Traffic_Level"]
        submitted = True
        st.info(
            f"Random sample: **{airline}** from **{source}** to **{destination}**, "
            f"{distance} miles, {weather} weather, {traffic} traffic."
        )

    # ---- Run Prediction ----
    if submitted:
        with st.spinner("Analyzing flight data and predicting..."):
            time.sleep(0.6)  # small delay for a smooth UX loading effect

            input_dict = {
                "Airline": safe_encode(encoders["Airline"], airline),
                "Source": safe_encode(encoders["Source"], source),
                "Destination": safe_encode(encoders["Destination"], destination),
                "Distance": distance,
                "Scheduled_Departure": dep_hour,
                "Scheduled_Arrival": arr_hour,
                "Day_of_Week": day_of_week,
                "Month": month,
                "Weather": safe_encode(encoders["Weather"], weather),
                "Traffic_Level": safe_encode(encoders["Traffic_Level"], traffic),
            }
            input_df = pd.DataFrame([input_dict])[feature_cols]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            delay_confidence = probability[1] * 100
            ontime_confidence = probability[0] * 100

        # ---- Result Card ----
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-card" style="background-color:{DANGER}22; border: 2px solid {DANGER};">
                    <h2 style="color:{DANGER};">⚠️ Flight Likely DELAYED</h2>
                    <p style="font-size:18px;">Confidence: <b>{delay_confidence:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="background-color:{SUCCESS}22; border: 2px solid {SUCCESS};">
                    <h2 style="color:{SUCCESS};">✅ Flight Likely ON TIME</h2>
                    <p style="font-size:18px;">Confidence: <b>{ontime_confidence:.1f}%</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Animated confidence progress bar ----
        st.markdown("#### Prediction Confidence")
        progress_bar = st.progress(0)
        target = int(delay_confidence if prediction == 1 else ontime_confidence)
        for pct in range(0, target + 1, 5):
            progress_bar.progress(min(pct, 100))
            time.sleep(0.01)
        progress_bar.progress(target)

        # ---- Probability chart ----
        prob_fig = go.Figure(
            data=[
                go.Bar(
                    x=["On Time", "Delayed"],
                    y=[probability[0] * 100, probability[1] * 100],
                    marker_color=[SUCCESS, DANGER],
                    text=[f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"],
                    textposition="auto",
                )
            ]
        )
        prob_fig.update_layout(
            title="Prediction Probability Breakdown",
            paper_bgcolor=CARD_COLOR,
            plot_bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR),
            yaxis_title="Probability (%)",
            height=350,
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # ---- Save to history ----
        st.session_state.history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Distance": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Prediction": "Delayed" if prediction == 1 else "On Time",
            "Confidence(%)": round(delay_confidence if prediction == 1 else ontime_confidence, 1),
        })

# ============================================================
# PAGE 2: DATASET INSIGHTS
# ============================================================
elif page == "📊 Dataset Insights":
    st.markdown("# 📊 Dataset Insights")
    st.markdown("Explore the historical flight dataset used to train the model.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown(f"**Total Records:** {len(df):,}  |  **Columns:** {len(df.columns)}")
    st.markdown("</div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🥧 Delay Distribution")
        delay_counts = df["Delay"].value_counts().rename({0: "On Time", 1: "Delayed"})
        pie_fig = px.pie(
            names=delay_counts.index,
            values=delay_counts.values,
            color=delay_counts.index,
            color_discrete_map={"On Time": SUCCESS, "Delayed": DANGER},
            hole=0.45,
        )
        pie_fig.update_layout(
            paper_bgcolor=CARD_COLOR, font=dict(color=TEXT_COLOR), height=350
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🌦️ Weather Distribution")
        weather_counts = df["Weather"].value_counts()
        bar_fig = px.bar(
            x=weather_counts.index,
            y=weather_counts.values,
            color=weather_counts.index,
            labels={"x": "Weather", "y": "Number of Flights"},
        )
        bar_fig.update_layout(
            paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR), showlegend=False, height=350
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📅 Monthly Delays")
        monthly = df.groupby("Month")["Delay"].mean().reset_index()
        monthly["Delay"] = monthly["Delay"] * 100
        month_fig = px.line(
            monthly, x="Month", y="Delay", markers=True,
            labels={"Delay": "Delay Rate (%)"},
        )
        month_fig.update_traces(line_color=PRIMARY)
        month_fig.update_layout(
            paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR), height=350
        )
        st.plotly_chart(month_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ✈️ Airline Delay Rates")
        airline_delay = df.groupby("Airline")["Delay"].mean().reset_index()
        airline_delay["Delay"] = airline_delay["Delay"] * 100
        airline_delay = airline_delay.sort_values("Delay", ascending=False)
        airline_fig = px.bar(
            airline_delay, x="Airline", y="Delay",
            color="Delay", color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            labels={"Delay": "Delay Rate (%)"},
        )
        airline_fig.update_layout(
            paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR), height=350
        )
        st.plotly_chart(airline_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "🧠 Model Performance":
    st.markdown("# 🧠 Model Performance")
    st.markdown("Evaluation metrics for the trained Random Forest Classifier.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Accuracy", f"{accuracy*100:.1f}%", SUCCESS)
    with m2:
        metric_card("Precision (Delayed)", f"{report_dict['1']['precision']*100:.1f}%", PRIMARY)
    with m3:
        metric_card("Recall (Delayed)", f"{report_dict['1']['recall']*100:.1f}%", WARNING)
    with m4:
        metric_card("F1-Score (Delayed)", f"{report_dict['1']['f1-score']*100:.1f}%", PRIMARY)

    st.markdown("<br>", unsafe_allow_html=True)
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🔢 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        fig.patch.set_facecolor(CARD_COLOR)
        ax.set_facecolor(CARD_COLOR)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["On Time", "Delayed"],
            yticklabels=["On Time", "Delayed"],
            ax=ax, cbar=False, annot_kws={"color": "black", "size": 13}
        )
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with perf_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols, "Importance": importances
        }).sort_values("Importance", ascending=True)
        imp_fig = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
        )
        imp_fig.update_layout(
            paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
            font=dict(color=TEXT_COLOR), height=350
        )
        st.plotly_chart(imp_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Classification Report")
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE 4: PREDICTION HISTORY
# ============================================================
elif page == "📁 Prediction History":
    st.markdown("# 📁 Prediction History")
    st.markdown("All predictions made during this session.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if len(st.session_state.history) == 0:
        st.info("No predictions made yet. Head to the **Predict Delay** page to get started!")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                label="⬇️ Download Prediction History (CSV)",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with col_b:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="custom-footer">
        Built with ❤️ using Python, Scikit-learn & Streamlit &nbsp;|&nbsp;
        Flight Delay Prediction System &copy; 2026
    </div>
    """,
    unsafe_allow_html=True,
)
