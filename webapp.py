# webapp.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bangladesh International Flight Fare Predictor Dashboard", layout="wide")

# --- CONFIG FILES ---
DB_PATH = "flight_fare.db"
MODEL_PATH = "Airline_rf_model.joblib"   

# --- FEATURES (exact order as your model) ---
FEATURE_ORDER = [
    "Base Fare (BDT)",
    "Tax & Surcharge (BDT)",
    "Aircraft_Label",
    "flight_mins",
    "Class_Label",
    "holiday_flag",
    "Seasonality_Label",
    "Destination_Label",
    "Departure_Minute",
    "Is_Night_Flight",
    "Airline_Label",
    "Booking_Label",
    "Source_Label",
    "Departure_Hour",
    "Is_Premium_Airline",
    "Departure_Month",
    "Days Before Departure",
    "Is_Direct"          # ⭐ NEW FEATURE ADDED ⭐
]

# --- UTIL: safe median helper ---
def safe_median(df, col, default=0, min_val=None):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        med = pd.to_numeric(df[col], errors="coerce").median()
        med = med if not np.isnan(med) else default
    else:
        med = default
    if min_val is not None:
        try:
            return int(max(med, min_val))
        except:
            return int(min_val)
    return med

# --- LOAD DB ---
@st.cache_data
def load_db(path=DB_PATH):
    if not os.path.exists(path):
        return pd.DataFrame(), f"DB not found: {path}"
    try:
        conn = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM flight_data", conn)
        conn.close()
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error reading flight_data: {e}"

df, db_err = load_db()

# --- HEADER ---
st.markdown("<div style='text-align:center; font-size:34px; font-weight:800;'>✈️ Bangladesh International Flight Fare Predictor Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:13px; color:#6b7280'>Created by Balram Shah — Flight Fare Prediction & Analysis</div>", unsafe_allow_html=True)
st.write("")

# DB error
if db_err:
    st.error(db_err)
    st.stop()

# quick preview
with st.expander("Preview: first 10 rows of flight_data"):
    st.dataframe(df.head(10))

# --- SIDEBAR FILTERS (for EDA) ---
st.sidebar.header("Filters & Prediction Inputs")
st.sidebar.markdown("Filter dataset and use the prediction panel below to estimate **Total Fare (BDT)**.")

def safe_unique(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

airlines = safe_unique("Airline")
sources = safe_unique("Source")
destinations = safe_unique("Destination")
classes = safe_unique("Class")

selected_airlines = st.sidebar.multiselect("Airline", airlines, default=airlines if airlines else None)
selected_sources = st.sidebar.multiselect("Source", sources, default=sources if sources else None)
selected_destinations = st.sidebar.multiselect("Destination", destinations, default=destinations if destinations else None)
selected_classes = st.sidebar.multiselect("Class", classes, default=classes if classes else None)

# numeric sliders
min_days = int(df["Days Before Departure"].min()) if "Days Before Departure" in df else 0
max_days = int(df["Days Before Departure"].max()) if "Days Before Departure" in df else 365
days_range = st.sidebar.slider("Days Before Departure", min_value=min_days, max_value=max_days, value=(min_days, max_days))

min_fm = int(df["flight_mins"].min()) if "flight_mins" in df else 10
max_fm = int(df["flight_mins"].max()) if "flight_mins" in df else 2000
flight_mins_range = st.sidebar.slider("Flight Duration (mins)", min_value=min_fm, max_value=max_fm, value=(min_fm, max_fm))

# apply filters
filtered = df.copy()
if selected_airlines:
    filtered = filtered[filtered["Airline"].isin(selected_airlines)]
if selected_sources:
    filtered = filtered[filtered["Source"].isin(selected_sources)]
if selected_destinations:
    filtered = filtered[filtered["Destination"].isin(selected_destinations)]
if selected_classes:
    filtered = filtered[filtered["Class"].isin(selected_classes)]
if "Days Before Departure" in filtered:
    filtered = filtered[(filtered["Days Before Departure"] >= days_range[0]) & (filtered["Days Before Departure"] <= days_range[1])]
if "flight_mins" in filtered:
    filtered = filtered[(filtered["flight_mins"] >= flight_mins_range[0]) & (filtered["flight_mins"] <= flight_mins_range[1])]

# --- KPIs ---
st.markdown("### Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Records (filtered)", f"{len(filtered):,}")
c2.metric("Avg Total Fare (BDT)", f"{filtered['Total Fare (BDT)'].mean():,.2f}" if "Total Fare (BDT)" in filtered else "N/A")
c3.metric("Avg Base Fare (BDT)", f"{filtered['Base Fare (BDT)'].mean():,.2f}" if "Base Fare (BDT)" in filtered else "N/A")

st.markdown("---")

# --- CHARTS (EDA) ---
# (Unchanged — same as your old version)

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Helper: mapping factorize ---
@st.cache_data
def build_mappings(df):
    mappings = {}
    cat_pairs = [
        ("Aircraft_Label", "Aircraft Type"),
        ("Class_Label", "Class"),
        ("Seasonality_Label", None),
        ("Destination_Label", "Destination"),
        ("Airline_Label", "Airline"),
        ("Booking_Label", "Booking Source"),
        ("Source_Label", "Source"),
    ]
    for lbl_col, raw_col in cat_pairs:
        if lbl_col in df.columns:
            mappings[lbl_col] = None
        elif raw_col and raw_col in df.columns:
            vals, codes = pd.factorize(df[raw_col].astype(str), sort=True)
            mappings[lbl_col] = {v: i for i, v in enumerate(vals)}
        else:
            mappings[lbl_col] = None
    return mappings

mappings = build_mappings(df)

# --- PREDICTION PANEL ---
with st.expander("🤖 Predict Total Fare (ML Model)"):

    with st.form("predict_form"):
        r1 = st.columns(4)
        base_fare = r1[0].number_input("Base Fare (BDT)", value=float(safe_median(df, "Base Fare (BDT)", 0, 0)))
        tax_surcharge = r1[1].number_input("Tax & Surcharge (BDT)", value=float(safe_median(df, "Tax & Surcharge (BDT)", 0, 0)))
        flight_mins = r1[2].number_input("Flight duration (mins)", min_value=10, max_value=2000, value=int(safe_median(df, "flight_mins", 300, 10)))
        days_before = r1[3].number_input("Days Before Departure", min_value=0, max_value=5000, value=int(safe_median(df, "Days Before Departure", 30, 0)))

        r2 = st.columns(4)
        aircraft_label = r2[0].number_input("Aircraft_Label", value=0)
        class_label = r2[1].number_input("Class_Label", value=0)
        season_label = r2[2].number_input("Seasonality_Label", value=0)
        dest_label = r2[3].number_input("Destination_Label", value=0)

        r3 = st.columns(4)
        airline_label = r3[0].number_input("Airline_Label", value=0)
        booking_label = r3[1].number_input("Booking_Label", value=0)
        source_label = r3[2].number_input("Source_Label", value=0)
        is_night = r3[3].selectbox("Is Night Flight?", options=[0,1], index=0)

        r4 = st.columns(4)
        dep_hour = r4[0].slider("Departure Hour", 0, 23, int(safe_median(df, "Departure_Hour", 10)))
        dep_min = r4[1].slider("Departure Minute", 0, 59, int(safe_median(df, "Departure_Minute", 0)))
        dep_month = r4[2].slider("Departure Month", 1, 12, int(safe_median(df, "Departure_Month", 6)))
        is_direct = r4[3].selectbox("Is Direct Flight?", options=[0,1], index=1)   # ⭐ NEW INPUT ⭐

        submitted = st.form_submit_button("Predict Total Fare")

    if submitted:
        record = {
            "Base Fare (BDT)": float(base_fare),
            "Tax & Surcharge (BDT)": float(tax_surcharge),
            "Aircraft_Label": int(aircraft_label),
            "flight_mins": int(flight_mins),
            "Class_Label": int(class_label),
            "holiday_flag": int(df["holiday_flag"].median()) if "holiday_flag" in df else 0,
            "Seasonality_Label": int(season_label),
            "Destination_Label": int(dest_label),
            "Departure_Minute": int(dep_min),
            "Is_Night_Flight": int(is_night),
            "Airline_Label": int(airline_label),
            "Booking_Label": int(booking_label),
            "Source_Label": int(source_label),
            "Departure_Hour": int(dep_hour),
            "Is_Premium_Airline": int(df["Is_Premium_Airline"].median()) if "Is_Premium_Airline" in df else 0,
            "Departure_Month": int(dep_month),
            "Days Before Departure": int(days_before),
            "Is_Direct": int(is_direct)        
        }

        input_df = pd.DataFrame([record])
        for c in FEATURE_ORDER:
            if c not in input_df:
                input_df[c] = 0
        input_df = input_df[FEATURE_ORDER]

        try:
            pred = model.predict(input_df)[0]
            st.success(f"✅ Estimated Total Fare: {pred:,.2f} BDT")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Data preview ---
st.markdown("---")
st.markdown("### Filtered Data Preview (top 200 rows)")
st.dataframe(filtered.head(200))

st.sidebar.download_button("⬇️ Download Filtered Data CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="filtered_flight_data.csv")

st.markdown("---")
st.markdown("© 2025 | Created by Balram Shah | Powered by Streamlit")

