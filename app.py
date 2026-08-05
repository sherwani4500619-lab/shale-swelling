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
    
    # Final swelling value for the summary display
    final_swelling = swelling[-1] if len(swelling) > 0 else 0.0

    # --- Display Computed Parameters ---
    st.markdown("### 📊 Final Output Summary")
    with st.info("Simulation Results:"):
        col1, col2 = st.columns(2)
        col1.metric("Total Time Taken", f"{max_time_input} {time_unit}")
        col2.metric("Final Predicted Swelling", f"{final_swelling:.4f} %")

    st.divider()

    # --- Prepare Data Frames ---
    
    # 1. Dataframe specifically for the Chart (Time in Hours)
    t_hours = t_sec / 3600.0
    chart_data = pd.DataFrame({
        "Time (in hrs)": t_hours,
        "Predicted Swelling": swelling
    }).set_index("Time (in hrs)")

    # 2. Dataframe specifically for CSV Export (Time in Seconds)
    results_df = pd.DataFrame({
        "Time (in sec)": t_sec,
        "Predicted Swelling": swelling
    })

    # --- Organize Outputs into Tabs ---
    tab1, tab2 = st.tabs(["📈 Interactive Swelling Chart", "💾 Export Data"])

    with tab1:
        st.line_chart(chart_data)

    with tab2:
        st.markdown("#### Download Results")
        st.write("Click the button below to download the full time-series data (Time in seconds vs. Predicted Swelling).")
        
        # Convert the export dataframe to CSV
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name="shale_swelling_results.csv",
            mime="text/csv"
        )
