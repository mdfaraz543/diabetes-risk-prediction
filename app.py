import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# LOAD MODEL (Pipeline with scaler included)
# -------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model

model = load_model()

# -------------------------------------------------
# CLEAN MODERN CSS (Neutral Colors + Animations)
# -------------------------------------------------
st.markdown("""
<style>

/* ===============================
   FULL APP WRAPPER
=============================== */

.stApp {
    background: linear-gradient(135deg, #eef2f7, #f8fafc);
    font-family: 'Segoe UI', sans-serif;
    animation: pageEnter 1s ease-out;
}

/* Smooth page load animation */
@keyframes pageEnter {
    0% {
        opacity: 0;
        transform: translateY(40px) scale(0.97);
    }
    100% {
        opacity: 1;
        transform: translateY(0px) scale(1);
    }
}

/* ===============================
   GRADIENT ANIMATED BORDER
=============================== */

.main > div {
    position: relative;
    border-radius: 40px;
    padding: 35px;
    background: white;
    z-index: 1;
}

.main > div::before {
    content: "";
    position: absolute;
    inset: -4px;
    border-radius: 45px;
    background: linear-gradient(
        45deg,
        #0A9BFB,
        #00C2FF,
        #7C3AED,
        #FF3B3B,
        #FFD60A,
        #0A9BFB
    );
    background-size: 400% 400%;
    animation: gradientMove 10s ease infinite;
    z-index: -1;
    filter: blur(8px);
    opacity: 0.7;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===============================
   TITLE (BIGGER + BOLDER)
=============================== */

h1 {
    text-align: center;
    font-size: 64px !important;
    font-weight: 900 !important;
    color: #111827 !important;
    margin-bottom: 35px;
    letter-spacing: 1px;
}

/* ===============================
   LABELS BIGGER
=============================== */

label {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1f2937 !important;
}

/* ===============================
   INPUT STYLING (PREMIUM NEUMORPHIC)
=============================== */

.stNumberInput input {
    border-radius: 18px !important;
    padding: 16px !important;
    font-size: 20px !important;
    border: none !important;
    background: linear-gradient(145deg, #ffffff, #f1f5f9) !important;
    box-shadow:
        inset 5px 5px 12px rgba(0,0,0,0.05),
        inset -5px -5px 12px rgba(255,255,255,0.9);
    transition: all 0.3s ease;
    font-weight: 600;
}

.stNumberInput input:focus {
    transform: scale(1.05);
    box-shadow:
        0 0 0 3px #0A9BFB,
        0 15px 40px rgba(10,155,251,0.35);
}

/* ===============================
   BUTTON PREMIUM (BIGGER)
=============================== */

.stButton>button {
    background: linear-gradient(135deg, #111827, #1f2937) !important;
    color: white !important;
    padding: 16px 45px !important;
    border-radius: 16px !important;
    font-size: 20px !important;
    font-weight: 800 !important;
    transition: all 0.35s ease;
    border: none !important;
}

.stButton>button:hover {
    transform: translateY(-6px) scale(1.05);
    box-shadow: 0 20px 45px rgba(0,0,0,0.3);
    background: linear-gradient(135deg, #0A9BFB, #2563eb) !important;
}

/* ===============================
   RESULT CARD (MORE PREMIUM)
=============================== */

.result-card {
    background: linear-gradient(145deg, #ffffff, #eef3f9);
    padding: 60px;
    border-radius: 32px;
    text-align: center;
    margin-top: 40px;
    box-shadow:
        30px 30px 80px rgba(0,0,0,0.15),
        -25px -25px 60px rgba(255,255,255,0.95);
    animation: fadeUp 0.8s ease;
    position: relative;
    overflow: hidden;
}

@keyframes fadeUp {
    from {opacity:0; transform:translateY(35px);}
    to {opacity:1; transform:translateY(0);}
}

.result-card h2 {
    font-size: 42px;
    font-weight: 900;
    margin-bottom: 15px;
    color: #111827;
}

.result-card p {
    font-size: 24px;
    font-weight: 600;
    color: #4b5563;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("<h1>🩺 Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Predict diabetes risk using patient health metrics</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left, right = st.columns([1.2, 1])

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
with left:

    st.markdown('<div class="patient-container">', unsafe_allow_html=True)

    st.subheader("📋 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col2:
        age = st.number_input("Age", 0, 120, 30)
        blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
        insulin = st.number_input("Insulin", 0, 900, 80)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    predict_btn = st.button("🔍 Predict Risk")

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
with right:
    st.subheader("🧠 Risk Intelligence")

    if predict_btn:

        input_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        input_df = pd.DataFrame([input_data])
        expected_columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        input_df = input_df[expected_columns]

        probability = model.predict_proba(input_df)[0][1] * 100

        if probability < 40:
            risk_label = "LOW RISK"
            main_color = "#78E393"
        elif probability < 70:
            risk_label = "MODERATE RISK"
            main_color = "#FFD60A"
        else:
            risk_label = "HIGH RISK"
            main_color = "#FF3B3B"

        # ===============================
        # ULTRA 3D GLASS CONTAINER
        # ===============================

        st.markdown(f"""
        <div style="
            position: relative;
            background: linear-gradient(145deg, #ffffff, #e9eef5);
            border-radius: 45px;
            box-shadow:
                25px 25px 60px rgba(0,0,0,0.18),
                -20px -20px 50px rgba(255,255,255,0.95),
                inset 0 8px 15px rgba(255,255,255,0.6);
            border: 1px solid rgba(255,255,255,0.4);
            margin-bottom: 30px;
        ">
            <div style="
                position:absolute;
                top:0;
                left:0;
                width:100%;
                height:50%;
                border-radius:45px 45px 0 0;
                background: linear-gradient(to bottom, rgba(255,255,255,0.8), transparent);
                pointer-events:none;
            ">
            </div>
        """, unsafe_allow_html=True)

        placeholder = st.empty()

        step = max(1, int(probability / 70))

        for val in range(0, int(probability) + 1, step):

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                number={
                    'suffix': "%",
                    'font': {'size': 64}
                },
                title={
                    'text': f"<b>{risk_label}</b>",
                    'font': {'size': 32}
                },
                gauge={
                    'shape': "angular",
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 2,
                        'tickcolor': "#9CA3AF"
                    },
                    'bar': {
                        'color': main_color,
                        'thickness': 0.6
                    },
                    'steps': [
                        {'range': [0, 40], 'color': "#38BDF8"},
                        {'range': [40, 70], 'color': "#FACC15"},
                        {'range': [70, 100], 'color': "#FB7185"}
                    ],
                }
            ))

            fig.update_layout(
                height=470,
                margin=dict(l=40, r=40, t=80, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'family': "Segoe UI"}
            )

            placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.006)

        # ===============================
        # Floating Glow Effect Below
        # ===============================

        st.markdown(f"""
        <div style="
            height:25px;
            width:70%;
            margin: -20px auto 20px auto;
            background: radial-gradient(circle, {main_color}55, transparent);
            filter: blur(20px);
            border-radius:50%;
        ">
        </div>
        """, unsafe_allow_html=True)

        # Premium Result Panel
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {main_color}35, white);
            border-radius: 28px;
            padding: 25px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.4);
        ">
            <h2 style="color:{main_color}; margin-bottom:8px;">
                {risk_label}
            </h2>
            <p style="margin:0; font-size:16px;">
                Advanced ML-based diabetes probability assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Enter patient details and click Predict Risk")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>"
    "⚠ For educational purposes only. Not a medical diagnosis."
    "</p>",
    unsafe_allow_html=True
)