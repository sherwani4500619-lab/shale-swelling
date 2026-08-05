import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Shale Swelling Model", 
    page_icon="🪨", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Header Section ---
st.title("🪨 Shale Swelling Prediction Model")
st.markdown("Calculate linear shale swelling percentage over time based on XRD mineralogy and Cation Exchange Capacity (CEC).")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.title("⚙️ Model Parameters")
st.sidebar.markdown("Adjust rock properties and simulation limits.")

with st.sidebar.expander("1. Cation Exchange Capacity", expanded=True):
    cec = st.number_input("CEC (meq/100 g)", value=25.0, step=0.5)

with st.sidebar.expander("2. XRD Mineralogy (wt.%)", expanded=True):
    sm = st.number_input("Smectite (Sm)", value=20.0, step=0.5)
    q = st.number_input("Quartz (Q)", value=35.0, step=0.5)
    dol = st.number_input("Dolomite (Dol)", value=10.0, step=0.5)
    cal = st.number_input("Calcite (Cal)", value=15.0, step=0.5)
    hal = st.number_input("Halite (Hal)", value=5.0, step=0.5)

with st.sidebar.expander("3. Time Settings", expanded=True):
    time_unit = st.selectbox("Input Time Unit", ["Hours", "Minutes", "Seconds", "Days"], index=0)
    max_time_input = st.number_input(f"Simulation Duration ({time_unit})", min_value=0.1, value=24.0, step=1.0)

# --- Unit Conversion ---
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
    st.error("⚠️ Error: The sum of non-swelling minerals (Q + Dol + Cal + Hal) cannot be zero.")
else:
    # 1. Mineralogy Factor R
    R = (cec * sm) / denominator

    # 2. Maximum Capacity A and Characteristic Time Tau
    A = 4.211172 + 0.915226 * (R ** 1.957788)
    tau = 1089.786186 + 4877.982915 * (R ** 1.096567)

    # 3. Time vector generation (Calculations strictly in SECONDS)
    t_sec = np.arange(0, max_time_sec + 1, 10)
    
    # 4. Dimensionless time x and Swelling S(t) (Using SECONDS)
    x = t_sec / tau
    
    exp_num = 0.565351
    exp_denom = 1.446685
    ratio_exp = exp_num / exp_denom

    swelling = A * (np.abs(x) ** exp_num) / ((1 + (np.abs(x) ** exp_denom)) ** ratio_exp)

    # 5. Convert time vector to HOURS strictly for the final output
    t_hours = t_sec / 3600.0

    # --- Display Computed Parameters ---
    st.markdown("### 📊 Model Outputs Summary")
    with st.info("Calculated constants based on current input parameters:"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Mineral Factor (R)", f"{R:.4f}")
        col2.metric("Max Capacity (A)", f"{A:.4f} %")
        col3.metric("Characteristic Time (τ)", f"{tau:.2f} s")

    st.divider()

    # --- Prepare Data Frame (Strictly limited to 2 output columns) ---
    results_df = pd.DataFrame({
        "time(in hrs)": t_hours,
        "Predicted swelling": swelling
    })

    # Set index for charting purposes so the x-axis reads correctly
    chart_data = results_df.set_index("time(in hrs)")

    # --- Organize Outputs into Tabs ---
    tab1, tab2 = st.tabs(["📈 Interactive Swelling Chart", "💾 Raw Data & Export"])

    with tab1:
        st.line_chart(chart_data)

    with tab2:
        st.markdown("#### Final Output Data")
        st.dataframe(results_df, use_container_width=True)
        
        # Convert the restricted dataframe to CSV
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name="shale_swelling_results.csv",
            mime="text/csv"
        )
