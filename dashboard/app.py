import sys
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API = "http://127.0.0.1:8000"
st.set_page_config(page_title="DSR Dashboard — Cledion", layout="wide")
st.title("⚡ Demand Side Response Dashboard")

# Sidebar: fetch sites
sites = []
try:
    r = requests.get(f"{API}/sites", timeout=5)
    r.raise_for_status()
    sites = r.json()["sites"]
except Exception:
    st.error("Cannot connect to API. Make sure the FastAPI server is running.")
    st.stop()
    sys.exit(1)

site = st.sidebar.selectbox("Site", sites)
date = st.sidebar.date_input("Date", value=pd.to_datetime("2018-07-10"))
run = st.sidebar.button("Run Simulation")


@st.cache_data(ttl=300)
def fetch_simulation(site_id, date_str):
    resp = requests.get(f"{API}/simulate", params={"site_id": site_id, "date": date_str}, timeout=60)
    resp.raise_for_status()
    return resp.json()


if run:
    date_str = date.strftime("%Y-%m-%d")
    with st.spinner("Running simulation... please wait ⏳"):
        try:
            data = fetch_simulation(site, date_str)
        except requests.RequestException as e:
            st.error(f"API error: {e.response.text if e.response is not None else str(e)}")
            st.stop()
    df = pd.DataFrame(data["intervals"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Daily Revenue", f"{data['daily_total_eur']:.4f} €")
    c2.metric("⚠️ Risk P10", f"{data['risk_p10_eur']:.4f} €")
    c3.metric("⚡ Activations", data["activated_intervals"])

    # Panel 1: Load + Price + Signal
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["load_kw"], name="Load (kW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price_eur_mwh"], name="Price (€/MWh)", line=dict(color="red")), secondary_y=True)
    for _, row in df[df["activate"] == 1].iterrows():
        fig.add_vrect(x0=row["timestamp"], x1=row["timestamp"] + pd.Timedelta(minutes=15), fillcolor="green", opacity=0.2, line_width=0)
    fig.update_layout(title="Load, Price & Activation Signal")
    fig.update_yaxes(title_text="Load (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Price (€/MWh)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Panel 2: Top 10 Activation Windows
    st.subheader("🏆 Top 10 Activation Windows")
    activated = df[df["activate"] == 1].sort_values("revenue_eur", ascending=False).head(10)
    st.dataframe(activated[["timestamp", "load_kw", "price_eur_mwh", "flexible_kw", "revenue_eur"]])

    # Panel 3: Revenue per interval
    activated_only = df[df["activate"] == 1]
    fig2 = go.Figure(go.Bar(x=activated_only["timestamp"], y=activated_only["revenue_eur"]))
    fig2.update_layout(title="Revenue per Activated Interval (€)", xaxis_title="Timestamp", yaxis_title="Revenue (€)")
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    # Run with: streamlit run dashboard/app.py
    pass
