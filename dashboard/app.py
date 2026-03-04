"""
DSR Dashboard — Cledion
Streamlit dashboard that connects to the FastAPI backend to visualise
Demand Side Response simulation results for a selected site and date.

Panels:
    1. Load (kW) + Price (€/MWh) + Activation Signal overlay
    2. Top 10 highest-revenue activation windows (table)
    3. Revenue per activated 15-minute interval (bar chart)

Run with:
    streamlit run dashboard/app.py
"""

import sys, os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revenue import simulate

st.set_page_config(page_title="DSR Dashboard — Cledion", layout="wide")
st.title("⚡ Demand Side Response Dashboard")

# Resolve exact data path (project root when run via: streamlit run dashboard/app.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PARQUET = os.path.join(_PROJECT_ROOT, "data", "processed", "processed_data.parquet")
st.write("**Data file path:**", _DATA_PARQUET)

try:
    df_sites = pd.read_parquet(_DATA_PARQUET)
    sites = sorted(df_sites["site_id"].unique().tolist())
except Exception:
    st.error("Cannot load processed data. Make sure data/processed/processed_data.parquet exists.")
    st.stop()

# Debug: first 10 rows and load_kw range (values should be in thousands of kW)
with st.expander("Data check — first 10 rows & load_kw range"):
    st.dataframe(df_sites.head(10))
    load_min, load_max = df_sites["load_kw"].min(), df_sites["load_kw"].max()
    st.write("**load_kw** — min:", load_min, "| max:", load_max)
    if load_max <= 100 or (df_sites["load_kw"].diff().dropna().abs().max() < 1e-6 and len(df_sites) > 1):
        st.warning("Load values look wrong (0–100 or flat). Check ingestion and that raw energy_dataset.csv is used without normalisation.")

# Sidebar controls: user selects a site and date, then clicks Run Simulation
site = st.sidebar.selectbox("Site", sites)
date = st.sidebar.date_input("Date", value=pd.to_datetime("2018-07-10"))
run = st.sidebar.button("Run Simulation")


# Cache simulation results for 5 minutes (ttl=300s) to avoid re-running
# the same heavy computation when the user interacts with the page
@st.cache_data(ttl=300)
def fetch_simulation(site_id, date_str):
    # Call simulate() directly — no HTTP request needed
    result = simulate(site_id, date_str)
    if result.empty:
        return None
    # Ensure only the selected date is returned (96 intervals per day)
    result = result[result["timestamp"].astype(str).str.startswith(date_str)]
    if result.empty:
        return None
    result = result.copy()
    result["timestamp"] = result["timestamp"].astype(str)
    return {
        "daily_total_eur": float(result["daily_total_eur"].iloc[0]),
        "risk_p10_eur": float(result["risk_p10_eur"].iloc[0]),
        "activated_intervals": int((result["activate"] == 1).sum()),
        "intervals": result.to_dict(orient="records"),
    }


# Only run the simulation when the user clicks the button
if run:
    date_str = date.strftime("%Y-%m-%d")
    with st.spinner("Running simulation... please wait ⏳"):
        data = fetch_simulation(site, date_str)
    if data is None:
        st.warning("No data found for this site and date.")
        st.stop()
    df = pd.DataFrame(data["intervals"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- KPI Summary Cards ---
    # Show daily revenue, risk metric (P10), and number of activated intervals
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Daily Revenue", f"{data['daily_total_eur']:.4f} €")
    c2.metric("⚠️ Risk P10", f"{data['risk_p10_eur']:.4f} €")
    c3.metric("⚡ Activations", data["activated_intervals"])

    # --- Panel 1: Load + Price + Activation Signal ---
    # Dual-axis chart: load (bar, left axis) and price (line, right axis)
    # Green shaded regions mark intervals where DSR activation was triggered
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["load_kw"], name="Load (kW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price_eur_mwh"], name="Price (€/MWh)", line=dict(color="red")), secondary_y=True)
    
    for _, row in df[df["activate"] == 1].iterrows():
        fig.add_vrect(x0=row["timestamp"], x1=row["timestamp"] + pd.Timedelta(minutes=15), fillcolor="green", opacity=0.2, line_width=0)
    fig.update_layout(
        title="Load, Price & Activation Signal",
        height=500,
        bargap=0.1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat="%H:%M", title="Time"),
    )
    fig.update_yaxes(title_text="Load (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Price (€/MWh)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Panel 2: Top 10 Activation Windows ---
    # Show the 10 intervals with the highest revenue when activation was triggered
    st.subheader("🏆 Top 10 Activation Windows")
    activated = df[df["activate"] == 1].sort_values("revenue_eur", ascending=False).head(10)
    st.dataframe(activated[["timestamp", "load_kw", "price_eur_mwh", "flexible_kw", "revenue_eur"]])

    # --- Panel 3: Revenue per Activated Interval ---
    # Bar chart showing how much revenue each activated interval generated
    activated_only = df[df["activate"] == 1]
    fig2 = go.Figure(go.Bar(x=activated_only["timestamp"], y=activated_only["revenue_eur"]))
    fig2.update_layout(title="Revenue per Activated Interval (€)", xaxis_title="Timestamp", yaxis_title="Revenue (€)")
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    # Run with: streamlit run dashboard/app.py
    pass
