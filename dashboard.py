"""Groundwater Level Monitoring & Prediction - Streamlit dashboard.

End-user facing application. No ML/experiment terminology, no model
comparison, no SHAP - the hybrid model runs invisibly in the background.
Reads only precomputed CSVs - no TensorFlow import in this process.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src import config

st.set_page_config(page_title="Groundwater Level Monitoring", layout="wide")

WELL_ID = config.WELL_IDS[0][0]
RESULTS_DIR = config.PROJECT_ROOT / "results"


@st.cache_data
def load_monthly():
    df = pd.read_csv(config.DATA_PROCESSED_DIR / f"{WELL_ID}_monthly.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_forecast():
    path = RESULTS_DIR / f"{WELL_ID}_next_month_forecast.csv"
    return pd.read_csv(path, parse_dates=["forecast_date", "current_date"]) if path.exists() else None


def change_at_lag(df, months_back):
    if len(df) <= months_back:
        return None
    return df.iloc[-1]["gwl"] - df.iloc[-1 - months_back]["gwl"]


def filter_by_range(df, label):
    if label == "All data":
        return df
    years = {"1 Year": 1, "5 Years": 5, "10 Years": 10}[label]
    cutoff = df["date"].max() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff]


def describe_change(change_ft, threshold=0.3):
    """Returns (status_label, emoji, plain-English sentence, short change
    phrase). Positive change = deeper water table = less favorable;
    negative = shallower = more favorable. Direction is always stated,
    even when the status is 'Stable' - only the magnitude determines the
    status label, not whether direction is shown."""
    if abs(change_ft) < 0.01:
        short = "No change"
        sentence = "No meaningful change is expected next month."
    elif change_ft > 0:
        short = f"{change_ft:.2f} ft deeper"
        sentence = f"The water table is expected to get {change_ft:.2f} ft deeper next month."
    else:
        short = f"{abs(change_ft):.2f} ft shallower"
        sentence = f"The water table is expected to get {abs(change_ft):.2f} ft shallower next month."

    if change_ft > threshold:
        status_label, emoji = "Declining", "🔴"
    elif change_ft < -threshold:
        status_label, emoji = "Recovering", "🟢"
    else:
        status_label, emoji = "Stable", "🟡"

    return status_label, emoji, sentence, short


monthly = load_monthly()
forecast = load_forecast()

st.title("🌊 Groundwater Level Monitoring & Prediction")
st.caption(f"USGS Well {WELL_ID} — California")
st.divider()

# ---------- 2. Current situation ----------
current_gwl = monthly.iloc[-1]["gwl"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Groundwater Level", f"{current_gwl:.2f} ft")

if forecast is not None:
    row = forecast.iloc[0]
    status_label, emoji, status_sentence, change_short = describe_change(row["change_ft"])

    c2.metric("Predicted Next Month", f"{row['predicted_gwl']:.2f} ft")
    c3.metric("Expected Change", change_short)
    c4.metric("Status", f"{emoji} {status_label}")
    st.caption(status_sentence)
else:
    c2.metric("Predicted Next Month", "—")
    c3.metric("Expected Change", "—")
    c4.metric("Status", "—")
    st.caption("Run `python forecast_next_month_hybrid.py` to generate a forecast.")

st.divider()

# ---------- 3. Main chart: groundwater trend + forecast ----------
st.subheader("Groundwater Level Trend")
range_choice = st.radio("Time range", ["1 Year", "5 Years", "10 Years", "All data"],
                         index=2, horizontal=True, label_visibility="collapsed")
plot_df = filter_by_range(monthly, range_choice)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_df["date"], plot_df["gwl"], color="steelblue", linewidth=1.8, label="Historical")


ax.invert_yaxis()
ax.set_ylabel("Depth to water (ft) — lower on chart = shallower water table")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# ---------- 4 & 5: Recent movement + outlook ----------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Recent Groundwater Movement")
    for label, months in [("1 month", 1), ("3 months", 3), ("6 months", 6), ("12 months", 12)]:
        val = change_at_lag(monthly, months)
        if val is not None:
            if abs(val) < 0.01:
                st.write(f"**{label}:** No change")
            else:
                direction = "deeper" if val > 0 else "shallower"
                st.write(f"**{label}:** {abs(val):.2f} ft {direction}")

with col_b:
    st.subheader("Next-Month Outlook")
    if forecast is not None:
        row = forecast.iloc[0]
        status_label, emoji, status_sentence, _ = describe_change(row["change_ft"])
        st.write(f"**Current:** {row['current_gwl']:.2f} ft")
        st.write(f"**Forecast:** {row['predicted_gwl']:.2f} ft")
        st.write(f"**{emoji} {status_label}** — {status_sentence}")
    else:
        st.write("No forecast available yet.")

st.divider()

# ---------- 6. Climate & groundwater ----------
st.subheader("Climate & Groundwater Conditions")
climate_range = st.radio("Climate time range", ["1 Year", "5 Years", "10 Years", "All data"],
                          index=1, horizontal=True, label_visibility="collapsed", key="climate_range")
climate_df = filter_by_range(monthly, climate_range)

fig2, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].bar(climate_df["date"], climate_df["rainfall_mm"], width=20, color="skyblue")
axes[0].set_ylabel("Rainfall (mm)")
axes[1].plot(climate_df["date"], climate_df["temp_c"], color="darkorange")
axes[1].set_ylabel("Temp (°C)")
plt.tight_layout()
st.pyplot(fig2)

st.divider()

# ---------- 7. Data explorer ----------
st.subheader("Explore Historical Data")
col1, col2 = st.columns([2, 1])
with col1:
    explore_range = st.selectbox("Date range", ["Last 1 Year", "Last 5 Years", "Last 10 Years", "All data"])
with col2:
    variable = st.selectbox("Variable", ["Groundwater Level", "Rainfall", "Temperature", "All"])

range_map = {"Last 1 Year": "1 Year", "Last 5 Years": "5 Years", "Last 10 Years": "10 Years", "All data": "All data"}
explore_df = filter_by_range(monthly, range_map[explore_range])

col_map = {"Groundwater Level": ["date", "gwl"], "Rainfall": ["date", "rainfall_mm"],
           "Temperature": ["date", "temp_c"], "All": ["date", "gwl", "rainfall_mm", "temp_c"]}
display_df = explore_df[col_map[variable]]

st.dataframe(display_df, use_container_width=True, height=250)
st.download_button("Download this data as CSV", display_df.to_csv(index=False),
                    file_name=f"{WELL_ID}_{variable.lower().replace(' ', '_')}.csv", mime="text/csv")

st.divider()
st.caption(
    "Forecast note: The next-month prediction is a genuine single-step forecast using the "
    "latest available groundwater history. Climate inputs for the forecast month use seasonal "
    "averages because actual future weather is unavailable."
)