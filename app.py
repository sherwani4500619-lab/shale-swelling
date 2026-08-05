import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="Shale Swelling Dashboard", layout="wide")

st.title("Shale Swelling Prediction Dashboard")
st.write("Calculate linear shale swelling percentage over time based on XRD mineralogy and CEC input parameters.")

# --- Sidebar Inputs ---
st.sidebar.header("1. Cation Exchange Capacity")
cec = st.sidebar.number_input("CEC (meq/100 g)", min_value=0.0, value=25.0, step=0.5)

st.sidebar.header("2. XRD Mineralogy (wt.%)")
sm = st.sidebar.number_input("Smectite (Sm)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
q = st.sidebar.number_input("Quartz (Q)", min_value=0.0, max_value=100.0, value=35.0, step=0.5)
dol = st.sidebar.number_input("Dolomite (Dol)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
cal = st.sidebar.number_input("Calcite (Cal)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
hal = st.sidebar.number_input("Halite (Hal)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

st.sidebar.header("3. Time Settings")
time_unit = st.sidebar.selectbox("Display Time Unit", ["Hours", "Minutes", "Seconds", "Days"], index=0)
max_time_input = st.sidebar.number_input(f"Simulation Duration ({time_unit})", min_value=0.1, value=24.0, step=1.0)
time_steps = st.sidebar.number_input("Number of Calculated Steps", min_value=10, max_value=2000, value=200, step=10)

# --- Unit Conversion to Seconds (Since equation requires t in seconds) ---
unit_to_seconds = {
    "Seconds": 1.0,
    "Minutes": 60.0,
    "Hours": 3600.0,
    "Days": 86400.0
}
max_time_sec = max_time_input * unit_to_seconds[time_unit]

# --- Core Mathematical Calculations ---
denominator = q + dol + cal + hal

if denominator == 0:
    st.error("Error: The sum of non-swelling minerals (Q + Dol + Cal + Hal) cannot be zero.")
else:
    # 1. Mineralogy Factor R
    R = (cec * sm) / denominator

    # 2. Maximum Capacity A and Characteristic Time Tau
    A = 4.211172 + 0.915226 * (R ** 1.957788)
    tau = 1089.786186 + 4877.982915 * (R ** 1.096567)

    # 3. Time vector generation
    t_sec = np.linspace(0, max_time_sec, int(time_steps))
    t_display = t_sec / unit_to_seconds[time_unit]

    # 4. Dimensionless time x and Swelling S(t)
    x = t_sec / tau
    
    exp_num = 0.565351
    exp_denom = 1.446685
    ratio_exp = exp_num / exp_denom

    swelling = A * (x ** exp_num) / ((1 + (x ** exp_denom)) ** ratio_exp)

    # --- Display Computed Parameters ---
    st.subheader("Model Outputs Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mineral Factor (R)", f"{R:.4f}")
    col2.metric("Max Capacity (A)", f"{A:.4f} %")
    col3.metric("Characteristic Time (τ)", f"{tau:.2f} s")

    st.markdown("---")

    # --- Plotting (Using Streamlit Native Charts instead of Matplotlib) ---
    st.subheader("Swelling vs. Time Curve")
    
    # Create a simple dataframe for the chart
    chart_data = pd.DataFrame({
        f"Time ({time_unit})": t_display,
        "Predicted Swelling S(t) (%)": swelling
    }).set_index(f"Time ({time_unit})")
    
    # Display the interactive line chart
    st.line_chart(chart_data)

    # --- Prepare Data Frame for Export ---
    results_df = pd.DataFrame({
        f"Time ({time_unit})": t_display,
        "Time (Seconds)": t_sec,
        "Dimensionless Time (x)": x,
        "Predicted Swelling S(t) (%)": swelling
    })

    # --- Excel File Export ---
    st.subheader("Data Export")

    def create_excel_report():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary Sheet
            inputs_summary = pd.DataFrame({
                "Parameter": [
                    "CEC (meq/100g)", "Smectite (wt.%)", "Quartz (wt.%)", 
                    "Dolomite (wt.%)", "Calcite (wt.%)", "Halite (wt.%)",
                    "Calculated R", "Predicted Max Capacity A (%)", "Characteristic Time Tau (s)"
                ],
                "Value": [cec, sm, q, dol, cal, hal, R, A, tau]
            })
            inputs_summary.to_excel(writer, sheet_name="Model Parameters", index=False)
            
            # Time-Series Sheet
            results_df.to_excel(writer, sheet_name="Swelling Profile Data", index=False)
            
        return output.getvalue()

    excel_bytes = create_excel_report()

    st.download_button(
        label="📥 Download Excel File (Inputs & Results)",
        data=excel_bytes,
        file_name="shale_swelling_model_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
