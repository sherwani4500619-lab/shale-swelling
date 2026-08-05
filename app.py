import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


UNIT_TO_SECONDS = {
    "Seconds": 1,
    "Minutes": 60,
    "Hours": 3600,
    "Days": 86400,
}


def compute_parameters(cec, smectite, quartz, dolomite, calcite, halite):
    denominator = quartz + dolomite + calcite + halite
    if denominator <= 0:
        raise ValueError("The sum of Quartz, Dolomite, Calcite, and Halite must be greater than 0.")

    r_value = (cec * smectite) / denominator
    a_value = 4.211172 + 0.915226 * (r_value ** 1.957788)
    tau_value = 1089.786186 + 4877.982915 * (r_value ** 1.096567)
    return r_value, a_value, tau_value


def compute_swelling_series(duration, unit, steps, a_value, tau_value):
    duration_seconds = duration * UNIT_TO_SECONDS[unit]
    times_seconds = np.linspace(0, duration_seconds, steps)
    x_value = times_seconds / tau_value

    swelling = a_value * (
        (x_value ** 0.565351)
        / ((1 + (x_value ** 1.446685)) ** (0.565351 / 1.446685))
    )
    times_display = times_seconds / UNIT_TO_SECONDS[unit]
    return times_display, times_seconds, swelling


def build_excel_file(inputs_df, results_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        results_df.to_excel(writer, sheet_name="Swelling Results", index=False)
    output.seek(0)
    return output


st.set_page_config(page_title="Shale Swelling Prediction Dashboard", layout="wide")
st.title("Shale Swelling Prediction Dashboard")

st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    cec = st.number_input("CEC", min_value=0.0, value=10.0, step=0.1)

with col2:
    st.markdown("**XRD Mineralogy (wt.%)**")
    smectite = st.number_input("Smectite", min_value=0.0, value=20.0, step=0.1)
    quartz = st.number_input("Quartz", min_value=0.0, value=30.0, step=0.1)
    dolomite = st.number_input("Dolomite", min_value=0.0, value=10.0, step=0.1)
    calcite = st.number_input("Calcite", min_value=0.0, value=10.0, step=0.1)
    halite = st.number_input("Halite", min_value=0.0, value=5.0, step=0.1)

with col3:
    st.markdown("**Time Settings**")
    duration = st.number_input("Simulation Duration", min_value=1.0, value=24.0, step=1.0)
    unit = st.selectbox("Time Unit", options=["Hours", "Minutes", "Seconds", "Days"])
    steps = st.number_input("Calculation Steps", min_value=2, value=100, step=1)

try:
    r_value, a_value, tau_value = compute_parameters(
        cec, smectite, quartz, dolomite, calcite, halite
    )

    time_display, time_seconds, swelling_series = compute_swelling_series(
        duration, unit, int(steps), a_value, tau_value
    )

    st.subheader("Calculated Model Parameters")
    m1, m2, m3 = st.columns(3)
    m1.metric("R (Mineralogical Ratio)", f"{r_value:.4f}")
    m2.metric("A (Max Capacity, %)", f"{a_value:.4f}")
    m3.metric("τ (Characteristic Time, s)", f"{tau_value:.2f}")

    st.subheader("Predicted Swelling vs Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_display, swelling_series, label="Predicted Swelling S(t)", color="blue")
    ax.axhline(y=a_value, color="red", linestyle="--", label="Max Capacity (A)")
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Predicted Linear Swelling (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    inputs_df = pd.DataFrame(
        {
            "Parameter": [
                "CEC",
                "Smectite",
                "Quartz",
                "Dolomite",
                "Calcite",
                "Halite",
                "Simulation Duration",
                "Time Unit",
                "Calculation Steps",
                "R",
                "A",
                "Tau (s)",
            ],
            "Value": [
                cec,
                smectite,
                quartz,
                dolomite,
                calcite,
                halite,
                duration,
                unit,
                int(steps),
                r_value,
                a_value,
                tau_value,
            ],
        }
    )

    results_df = pd.DataFrame(
        {
            f"Time ({unit})": time_display,
            "Time (s)": time_seconds,
            "Predicted Swelling (%)": swelling_series,
        }
    )

    excel_file = build_excel_file(inputs_df, results_df)

    st.download_button(
        label="Download Results as Excel (.xlsx)",
        data=excel_file,
        file_name="shale_swelling_prediction.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except ValueError as error:
    st.error(str(error))
