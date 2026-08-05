import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Shale Swelling Prediction & Mineralogy Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Shale Swelling Prediction, Optimization & Mineralogy Dashboard")
st.markdown("Upload lab swell meter data (`.csv` or `.xlsx`), select models to run, input XRD/CEC data, and export your results.")

# --- SIDEBAR: FILE UPLOAD & CONFIGURATION ---
st.sidebar.markdown("### 1. Upload Lab Data")
st.sidebar.caption("Upload a CSV or Excel file. Column 1 must be Time (seconds), Column 2 must be Swelling (%).")
uploaded_file = st.sidebar.file_uploader("Upload Experimental Data", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("👋 Please upload your experimental data (CSV or Excel) in the sidebar to run the auto-solver.")
    st.stop()

# --- DATA PARSING & CLEANING ---
try:
    if uploaded_file.name.endswith('.csv'):
        lab_data = pd.read_csv(uploaded_file)
        sheet_name_display = "CSV Dataset"
        
        lab_data.columns = lab_data.columns.astype(str).str.strip()
        raw_time = lab_data.iloc[:, 0]
        s_exp = pd.to_numeric(lab_data.iloc[:, 1], errors='coerce').values
    else:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Select Cell Sheet", excel_file.sheet_names)
        sheet_name_display = f"Sheet: {sheet_name}"
        
        lab_data = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=14)
        lab_data.columns = lab_data.columns.astype(str).str.strip()
        
        raw_time = lab_data.iloc[:, 1]
        s_exp = pd.to_numeric(lab_data.iloc[:, 2], errors='coerce').values

    def time_to_sec(t_str):
        try:
            if isinstance(t_str, (int, float)):
                return float(t_str)
            parts = str(t_str).split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(t_str)
        except:
            return np.nan

    t_sec = np.array([time_to_sec(t) for t in raw_time])
    valid_idx = ~np.isnan(t_sec) & ~np.isnan(s_exp)
    t_exp = t_sec[valid_idx]
    s_exp = s_exp[valid_idx]

    st.success(f"Successfully loaded data from **{sheet_name_display}** ({len(t_exp)} data points).")

except Exception as e:
    st.error(f"Error reading data file: {e}")
    st.stop()

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Selection")
all_models = ['Empirical', 'Custom Asymptotic', 'Higuchi', 'Korsmeyer-Peppas']
selected_models = st.sidebar.multiselect(
    "Choose models to evaluate:",
    options=all_models,
    default=all_models
)

# --- SIDEBAR: XRD & CEC PARAMETERS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Mineralogy (XRD & CEC)")
st.sidebar.caption("Input sample mineral properties for empirical swelling correlation.")
smectite_pct = st.sidebar.number_input("Smectite / Swelling Clay (%)", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
cec_val = st.sidebar.number_input("Cation Exchange Capacity (meq/100g)", min_value=0.0, max_value=150.0, value=25.0, step=0.5)

estimated_max_swell = (0.18 * smectite_pct) + (0.12 * cec_val)

# --- MATHEMATICAL MODEL DEFINITIONS ---
def model_empirical(t, a, b):
    return a * (1.0 - np.exp(-b * t))

def model_custom(t, a, b, c):
    return a * (1.0 - np.exp(-b * (t**c)))

def model_higuchi(t, k):
    return k * np.sqrt(np.maximum(t, 0))

def model_korsmeyer(t, k, n):
    return k * (np.maximum(t, 0) ** n)

# --- EXECUTION BUTTON ---
st.markdown("### ⚙️ Model Optimization & Statistical Evaluation")

if st.button("🚀 Run Instant Solver & Full Analysis", type="primary"):
    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar dropdown.")
    else:
        with st.spinner("Optimizing selected models and calculating performance metrics..."):
            
            step = max(1, len(t_exp) // 300)
            t_fit_x = t_exp[::step]
            s_fit_y = s_exp[::step]

            results = {}
            metrics_list = []

            models_dict = {
                'Empirical': (model_empirical, [max(s_fit_y), 1e-5]),
                'Custom Asymptotic': (model_custom, [max(s_fit_y), 1e-5, 1.0]),
                'Higuchi': (model_higuchi, [1.0]),
                'Korsmeyer-Peppas': (model_korsmeyer, [1.0, 0.5])
            }

            for name in selected_models:
                func, p0 = models_dict[name]
                try:
                    if name == 'Empirical':
                        popt, _ = curve_fit(func, t_fit_x, s_fit_y, p0=p0, maxfev=5000)
                    elif name == 'Custom Asymptotic':
                        popt, _ = curve_fit(func, t_fit_x, s_fit_y, p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, 5]), maxfev=5000)
                    elif name == 'Higuchi':
                        popt, _ = curve_fit(func, t_fit_x, s_fit_y, p0=p0, maxfev=5000)
                    elif name == 'Korsmeyer-Peppas':
                        popt, _ = curve_fit(func, t_fit_x, s_fit_y, p0=p0, bounds=([0, 0], [np.inf, 2]), maxfev=5000)

                    y_pred_full = func(t_exp, *popt)
                    results[name] = y_pred_full

                    rmse = np.sqrt(mean_squared_error(s_exp, y_pred_full))
                    r2 = r2_score(s_exp, y_pred_full)

                    metrics_list.append({
                        "Model": name,
                        "RMSE (%)": round(rmse, 4),
                        "R² Score": round(r2, 4)
                    })

                except Exception as e:
                    st.warning(f"Model {name} optimization skipped: {e}")

            st.session_state['optimization_results'] = results
            st.session_state['metrics_df'] = pd.DataFrame(metrics_list)
            st.session_state['t_exp'] = t_exp
            st.session_state['s_exp'] = s_exp
            st.session_state['xrd_cec_swell'] = estimated_max_swell
            
            st.success("✅ Optimization and background full-file analysis completed successfully!")

# --- DISPLAY RESULTS, CHARTS & METRICS ---
if 'optimization_results' in st.session_state:
    
    st.info(f"🔬 **Mineralogical Swelling Potential (XRD & CEC Correlation):** Estimated Max Swelling = **{st.session_state['xrd_cec_swell']:.2f}%** (Based on {smectite_pct}% Smectite and CEC of {cec_val} meq/100g)")

    st.markdown("#### 📈 Swelling vs. Time Model Comparison")
    
    chart_df = pd.DataFrame({
        "Time (seconds)": st.session_state['t_exp'],
        "Experimental Data": st.session_state['s_exp']
    })
    
    for model_name, preds in st.session_state['optimization_results'].items():
        chart_df[model_name] = preds
        
    melted_df = chart_df.melt(
        id_vars=["Time (seconds)"], 
        var_name="Legend / Model", 
        value_name="Swelling (%)"
    )
    
    line_chart = alt.Chart(melted_df).mark_line(strokeWidth=2).encode(
        x=alt.X('Time (seconds):Q', title='Time (seconds)'),
        y=alt.Y('Swelling (%):Q', title='Swelling (%)'),
        color=alt.Color('Legend / Model:N', title='Models & Data'),
        tooltip=['Time (seconds):Q', 'Swelling (%):Q', 'Legend / Model:N']
    ).properties(
        width=700,
        height=450
    ).interactive()
    
    st.altair_chart(line_chart, use_container_width=True)

    st.markdown("#### 📊 Statistical Performance Metrics")
    st.dataframe(st.session_state['metrics_df'], use_container_width=True)

    # --- EXCEL EXPORT FUNCTIONALITY ---
    st.markdown("### 💾 Export Results")
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        chart_df.to_excel(writer, sheet_name='Model Predictions', index=False)
        st.session_state['metrics_df'].to_excel(writer, sheet_name='Performance Metrics', index=False)
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Complete Results as Excel (.xlsx)",
        data=excel_data,
        file_name="shale_swelling_analysis_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
