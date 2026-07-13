import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Load Saved Model and Scaler
# -------------------------------------------------------

model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Oil & Gas Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    font-size:42px;
    color:#003366;
    font-weight:bold;
}

.subtitle{
    font-size:20px;
    color:#444444;
}

.stButton>button{
    background-color:#0E6BA8;
    color:white;
    font-size:18px;
    height:55px;
    width:100%;
    border-radius:10px;
}

.stButton>button:hover{
    background-color:#084C7F;
}

.result-good{
    background:#d4edda;
    padding:18px;
    border-radius:10px;
    color:#155724;
    font-size:22px;
    font-weight:bold;
}

.result-bad{
    background:#f8d7da;
    padding:18px;
    border-radius:10px;
    color:#721c24;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.markdown(
"<p class='title'>🏭 Oil & Gas Predictive Maintenance System</p>",
unsafe_allow_html=True
)

st.markdown(
"<p class='subtitle'>Predict industrial equipment failure using Machine Learning.</p>",
unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Equipment Details")

equipment = st.sidebar.selectbox(
    "Equipment Type",
    ["Pump", "Turbine", "Valve"]
)

# -------------------------------------------------------
# Input Columns
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        value=75.19
    )

    pressure = st.number_input(
        "Pressure (bar)",
        value=39.99
    )

    vibration = st.number_input(
        "Vibration (mm/s)",
        value=2.98
    )

    flow_rate = st.number_input(
        "Flow Rate (m³/h)",
        value=120.23
    )

    motor_current = st.number_input(
        "Motor Current (A)",
        value=89.67
    )

    oil_level = st.number_input(
        "Oil Level (%)",
        value=59.6
    )

with col2:

    runtime = st.number_input(
        "Runtime Hours",
        value=8707
    )

    ambient_temp = st.number_input(
        "Ambient Temperature (°C)",
        value=28.6
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=60.2
    )

    maintenance_days = st.number_input(
        "Maintenance Days",
        value=182
    )

    previous_failures = st.number_input(
        "Previous Failures",
        value=1
    )

# -------------------------------------------------------
# One-Hot Encoding
# -------------------------------------------------------

pump = 0
turbine = 0
valve = 0

if equipment == "Pump":
    pump = 1
elif equipment == "Turbine":
    turbine = 1
else:
    valve = 1

# -------------------------------------------------------
# Predict Button
# -------------------------------------------------------

if st.button("🔍 Predict Equipment Failure"):

    input_data = pd.DataFrame({

        "temperature_C":[temperature],
        "pressure_bar":[pressure],
        "vibration_mm_s":[vibration],
        "flow_rate_m3_h":[flow_rate],
        "motor_current_A":[motor_current],
        "oil_level_pct":[oil_level],
        "runtime_hours":[runtime],
        "ambient_temp_C":[ambient_temp],
        "humidity_pct":[humidity],
        "maintenance_days":[maintenance_days],
        "previous_failures":[previous_failures],
        "equipment_type_Pump":[pump],
        "equipment_type_Turbine":[turbine],
        "equipment_type_Valve":[valve]

    })

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    fail_probability = probability[0][1] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.markdown(
            "<div class='result-bad'>⚠ Equipment Likely to Fail</div>",
            unsafe_allow_html=True
        )

        st.error("Immediate maintenance is recommended.")

    else:

        st.markdown(
            "<div class='result-good'>✅ Equipment is Healthy</div>",
            unsafe_allow_html=True
        )

        st.success("No immediate maintenance required.")

    st.write("### Failure Probability")

    st.progress(float(fail_probability/100))

    st.metric(
        label="Failure Probability",
        value=f"{fail_probability:.2f}%"
    )

    st.divider()

    st.subheader("Equipment Summary")

    summary = pd.DataFrame({

        "Feature":[
            "Equipment",
            "Temperature",
            "Pressure",
            "Vibration",
            "Flow Rate",
            "Motor Current",
            "Oil Level",
            "Runtime",
            "Ambient Temperature",
            "Humidity",
            "Maintenance Days",
            "Previous Failures"
        ],

        "Value":[
            equipment,
            temperature,
            pressure,
            vibration,
            flow_rate,
            motor_current,
            oil_level,
            runtime,
            ambient_temp,
            humidity,
            maintenance_days,
            previous_failures
        ]

    })

    st.dataframe(summary, use_container_width=True)

st.divider()

st.caption("Developed by Ayush Manral | Machine Learning Predictive Maintenance Project")