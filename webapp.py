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
]

# --- UTIL: safe median helper (never lower than min_val) ---
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

# numeric sliders with safe bounds
min_days = int(df["Days Before Departure"].min()) if "Days Before Departure" in df.columns and pd.api.types.is_numeric_dtype(df["Days Before Departure"]) else 0
max_days = int(df["Days Before Departure"].max()) if "Days Before Departure" in df.columns and pd.api.types.is_numeric_dtype(df["Days Before Departure"]) else 365
days_range = st.sidebar.slider("Days Before Departure", min_value=min_days, max_value=max_days, value=(min_days, max_days))

min_fm = int(df["flight_mins"].min()) if "flight_mins" in df.columns and pd.api.types.is_numeric_dtype(df["flight_mins"]) else 10
max_fm = int(df["flight_mins"].max()) if "flight_mins" in df.columns and pd.api.types.is_numeric_dtype(df["flight_mins"]) else 2000
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
if "Days Before Departure" in filtered.columns:
    filtered = filtered[(filtered["Days Before Departure"] >= days_range[0]) & (filtered["Days Before Departure"] <= days_range[1])]
if "flight_mins" in filtered.columns:
    filtered = filtered[(filtered["flight_mins"] >= flight_mins_range[0]) & (filtered["flight_mins"] <= flight_mins_range[1])]

# --- KPIs ---
st.markdown("### Key Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Records (filtered)", f"{len(filtered):,}")
if "Total Fare (BDT)" in filtered.columns:
    c2.metric("Avg Total Fare (BDT)", f"{filtered['Total Fare (BDT)'].mean():,.2f}")
else:
    c2.metric("Avg Total Fare (BDT)", "N/A")
if "Base Fare (BDT)" in filtered.columns:
    c3.metric("Avg Base Fare (BDT)", f"{filtered['Base Fare (BDT)'].mean():,.2f}")
else:
    c3.metric("Avg Base Fare (BDT)", "N/A")

st.markdown("---")

# --- CHARTS (EDA) ---
st.markdown("## Insights & Charts")
theme = "plotly_white"
if "Total Fare (BDT)" in df.columns and "Departure_Month" in df.columns:
    with st.expander("Avg Total Fare by Departure Month"):
        agg = df.groupby("Departure_Month")["Total Fare (BDT)"].mean().reset_index()
        fig = px.line(agg, x="Departure_Month", y="Total Fare (BDT)", title="Avg Total Fare by Departure Month", template=theme)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Airline-wise Avg Total Fare"):
    if "Airline" in df.columns and "Total Fare (BDT)" in df.columns:
        agg = df.groupby("Airline")["Total Fare (BDT)"].mean().reset_index().sort_values("Total Fare (BDT)", ascending=False)
        fig = px.bar(agg, x="Airline", y="Total Fare (BDT)", title="Avg Total Fare by Airline", template=theme)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Airline or Total Fare column missing.")

with st.expander("Base Fare vs Tax & Surcharge"):
    if {"Base Fare (BDT)", "Tax & Surcharge (BDT)"}.issubset(df.columns):
        sample = df.sample(n=min(500, len(df)))
        fig = px.scatter(sample, x="Base Fare (BDT)", y="Tax & Surcharge (BDT)", size="Total Fare (BDT)" if "Total Fare (BDT)" in df.columns else None, title="Base vs Tax", template=theme)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Total Fare distribution"):
    if "Total Fare (BDT)" in df.columns:
        st.write(df["Total Fare (BDT)"].describe())
        fig = px.histogram(df.sample(n=min(1000, len(df))), x="Total Fare (BDT)", nbins=60, title="Total Fare Distribution", template=theme)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Correlation heatmap (numeric)"):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation.")

st.markdown("---")

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place your trained model file in the same folder as webapp.py.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- Helper: mapping factorize from DB for raw categorical -> label int (best-effort) ---
@st.cache_data
def build_mappings(df):
    mappings = {}
    # if label columns exist, no need to build mapping for them
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
            # mapping value->value (numeric labels already present)
            mappings[lbl_col] = None
        elif raw_col and raw_col in df.columns:
            vals, codes = pd.factorize(df[raw_col].astype(str), sort=True)
            # pd.factorize returns unique values as 'vals', but we want mapping raw->code
            mapping = {v: i for i, v in enumerate(vals)}
            mappings[lbl_col] = mapping
        else:
            mappings[lbl_col] = None
    return mappings

mappings = build_mappings(df)

# --- PREDICTION PANEL (form) ---
with st.expander("🤖 Predict Total Fare (ML Model)"):
    st.markdown("Enter flight details to predict **Total Fare (BDT)**. The app will prefer numeric label columns if present; otherwise it will map raw categories using DB-derived labels.")
    with st.form("predict_form"):
        # Row 1 — numeric basic
        r1 = st.columns(4)
        base_fare_default = safe_median(df, "Base Fare (BDT)", default=0, min_val=0)
        tax_default = safe_median(df, "Tax & Surcharge (BDT)", default=0, min_val=0)
        flight_mins_default = safe_median(df, "flight_mins", default=300, min_val=10)
        days_before_default = safe_median(df, "Days Before Departure", default=30, min_val=0)

        base_fare = r1[0].number_input("Base Fare (BDT)", value=float(base_fare_default), step=100.0, format="%.2f")
        tax_surcharge = r1[1].number_input("Tax & Surcharge (BDT)", value=float(tax_default), step=50.0, format="%.2f")
        flight_mins = r1[2].number_input("Flight duration (mins)", min_value=10, max_value=2000, value=int(flight_mins_default))
        days_before = r1[3].number_input("Days Before Departure", min_value=0, max_value=5000, value=int(days_before_default))

        # Row 2 — categorical labels or mapped raw
        r2 = st.columns(4)
        # Aircraft_Label
        if "Aircraft_Label" in df.columns:
            aircraft_label = r2[0].selectbox("Aircraft_Label", sorted(df["Aircraft_Label"].dropna().unique().tolist()))
        elif "Aircraft Type" in df.columns:
            aircraft_choice = r2[0].selectbox("Aircraft Type (raw)", sorted(df["Aircraft Type"].dropna().unique().astype(str).tolist()))
            aircraft_label = mappings.get("Aircraft_Label", {}).get(aircraft_choice, 0)
        else:
            aircraft_label = r2[0].number_input("Aircraft_Label (numeric)", min_value=0, value=0)

        # Class_Label
        if "Class_Label" in df.columns:
            class_label = r2[1].selectbox("Class_Label", sorted(df["Class_Label"].dropna().unique().tolist()))
        elif "Class" in df.columns:
            class_choice = r2[1].selectbox("Class (raw)", sorted(df["Class"].dropna().unique().astype(str).tolist()))
            class_label = mappings.get("Class_Label", {}).get(class_choice, 0)
        else:
            class_label = r2[1].number_input("Class_Label (numeric)", min_value=0, value=0)

        # Seasonality_Label
        if "Seasonality_Label" in df.columns:
            season_label = r2[2].selectbox("Seasonality_Label", sorted(df["Seasonality_Label"].dropna().unique().tolist()))
        else:
            season_label = r2[2].number_input("Seasonality (numeric)", value=0)

        # Destination_Label
        if "Destination_Label" in df.columns:
            dest_label = r2[3].selectbox("Destination_Label", sorted(df["Destination_Label"].dropna().unique().tolist()))
        elif "Destination" in df.columns:
            dest_choice = r2[3].selectbox("Destination (raw)", sorted(df["Destination"].dropna().unique().astype(str).tolist()))
            dest_label = mappings.get("Destination_Label", {}).get(dest_choice, 0)
        else:
            dest_label = r2[3].text_input("Destination (raw)", value="")

        # Row 3 — airline/booking/source flags
        r3 = st.columns(4)
        if "Airline_Label" in df.columns:
            airline_label = r3[0].selectbox("Airline_Label", sorted(df["Airline_Label"].dropna().unique().tolist()))
        elif "Airline" in df.columns:
            airline_choice = r3[0].selectbox("Airline (raw)", sorted(df["Airline"].dropna().unique().astype(str).tolist()))
            airline_label = mappings.get("Airline_Label", {}).get(airline_choice, 0)
        else:
            airline_label = r3[0].text_input("Airline (raw)", value="")

        if "Booking_Label" in df.columns:
            booking_label = r3[1].selectbox("Booking_Label", sorted(df["Booking_Label"].dropna().unique().tolist()))
        elif "Booking Source" in df.columns:
            booking_choice = r3[1].selectbox("Booking Source (raw)", sorted(df["Booking Source"].dropna().unique().astype(str).tolist()))
            booking_label = mappings.get("Booking_Label", {}).get(booking_choice, 0)
        else:
            booking_label = r3[1].number_input("Booking_Label (numeric)", min_value=0, value=0)

        if "Source_Label" in df.columns:
            source_label = r3[2].selectbox("Source_Label", sorted(df["Source_Label"].dropna().unique().tolist()))
        elif "Source" in df.columns:
            source_choice = r3[2].selectbox("Source (raw)", sorted(df["Source"].dropna().unique().astype(str).tolist()))
            source_label = mappings.get("Source_Label", {}).get(source_choice, 0)
        else:
            source_label = r3[2].text_input("Source (raw)", value="")

        # Is_Night_Flight
        if "Is_Night_Flight" in df.columns:
            is_night = r3[3].selectbox("Is Night Flight?", options=[0, 1], index=0, format_func=lambda x: "No" if x == 0 else "Yes")
        else:
            is_night = r3[3].selectbox("Is Night Flight?", options=[0, 1], index=0)

        # Row 4 — time & flags
        r4 = st.columns(4)
        dep_hour = r4[0].slider("Departure Hour", 0, 23, int(safe_median(df, "Departure_Hour", default=10, min_val=0)))
        dep_min = r4[1].slider("Departure Minute", 0, 59, int(safe_median(df, "Departure_Minute", default=0, min_val=0)))
        dep_month = r4[2].slider("Departure Month", 1, 12, int(safe_median(df, "Departure_Month", default=6, min_val=1)))
        is_direct = r4[3].selectbox("Is Direct Flight?", options=[0, 1], index=1, format_func=lambda x: "No" if x == 0 else "Yes")

        # submit
        submitted = st.form_submit_button("Predict Total Fare")

    # end form
    if submitted:
        # Build input in EXACT FEATURE ORDER (model expects this order)
        record = {}
        # numeric features
        record["Base Fare (BDT)"] = float(base_fare)
        record["Tax & Surcharge (BDT)"] = float(tax_surcharge)
        record["Aircraft_Label"] = int(aircraft_label) if isinstance(aircraft_label, (int, np.integer)) or str(aircraft_label).isdigit() else 0
        record["flight_mins"] = int(flight_mins)
        record["Class_Label"] = int(class_label) if isinstance(class_label, (int, np.integer)) or str(class_label).isdigit() else 0
        record["holiday_flag"] = int(0) if "holiday_flag" not in df.columns else int(df["holiday_flag"].median())  # default 0 or DB median
        record["Seasonality_Label"] = int(season_label) if isinstance(season_label, (int, np.integer)) or str(season_label).isdigit() else 0
        # destination label
        record["Destination_Label"] = int(dest_label) if isinstance(dest_label, (int, np.integer)) or str(dest_label).isdigit() else 0
        record["Departure_Minute"] = int(dep_min)
        record["Is_Night_Flight"] = int(is_night)
        record["Airline_Label"] = int(airline_label) if isinstance(airline_label, (int, np.integer)) or str(airline_label).isdigit() else 0
        record["Booking_Label"] = int(booking_label) if isinstance(booking_label, (int, np.integer)) or str(booking_label).isdigit() else 0
        record["Source_Label"] = int(source_label) if isinstance(source_label, (int, np.integer)) or str(source_label).isdigit() else 0
        record["Departure_Hour"] = int(dep_hour)
        record["Is_Premium_Airline"] = int(0) if "Is_Premium_Airline" not in df.columns else int(df["Is_Premium_Airline"].median())
        record["Departure_Month"] = int(dep_month)
        record["Days Before Departure"] = int(days_before)

        # create dataframe in right column order, fill missing with 0
        input_df = pd.DataFrame([record])
        for c in FEATURE_ORDER:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[FEATURE_ORDER]

        # Predict
        try:
            pred = model.predict(input_df)[0]
            st.success(f"✅ Estimated Total Fare: {pred:,.2f} BDT")
            # log to DB (optional)
            try:
                conn = sqlite3.connect(DB_PATH)
                log = input_df.copy()
                log["Predicted_Total_Fare_BDT"] = float(pred)
                log.to_sql("prediction_logs", conn, if_exists="append", index=False)
                conn.close()
            except Exception as e:
                st.warning(f"Prediction OK but logging failed: {e}")
        except Exception as e:
            st.error(f"Model prediction failed: {e}\nNote: Make sure the model expects the exact numeric/label features in the order above.")

# --- Data preview & download ---
st.markdown("---")
st.markdown("### Filtered Data Preview (top 200 rows)")
st.dataframe(filtered.head(200))

st.sidebar.download_button("⬇️ Download Filtered Data CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="filtered_flight_data.csv")

st.markdown("---")
st.markdown("© 2025 | Created by Balram Shah | Powered by Streamlit")
