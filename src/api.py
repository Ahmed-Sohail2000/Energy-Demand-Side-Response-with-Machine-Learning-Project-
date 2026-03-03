# pip install fastapi uvicorn
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from src.revenue import simulate
from src.activation import run as activation_run

app = FastAPI(title="DSR Simulation API")


@app.get("/")
def root():
    """Redirect users to docs; avoids 404 when opening base URL in browser."""
    return {
        "message": "DSR Simulation API",
        "docs": "/docs",
        "health": "/health",
        "sites": "/sites",
        "signal": "/signal?site_id=...&date=YYYY-MM-DD",
        "simulate": "/simulate?site_id=...&date=YYYY-MM-DD",
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/sites")
def sites():
    try:
        df = pd.read_parquet("data/processed/processed_data.parquet")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Processed data not found")
    return {"sites": sorted(df["site_id"].unique().tolist())}


@app.get("/signal")
def signal(
    site_id: str = Query(...),
    date: str = Query(...),
):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD")
    result = activation_run(site_id, date)
    if result.empty:
        raise HTTPException(status_code=404, detail="No signal found for given site_id and date")
    result = result.copy()
    result["timestamp"] = result["timestamp"].astype(str)
    return {"site_id": site_id, "date": date, "signal": result.to_dict(orient="records")}


@app.get("/simulate")
def simulate_endpoint(
    site_id: str = Query(...),
    date: str = Query(...),
):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD")
    result = simulate(site_id, date)
    if result.empty:
        raise HTTPException(status_code=404, detail="No data found for given site_id and date")
    result = result.copy()
    result["timestamp"] = result["timestamp"].astype(str)
    return {
        "site_id": site_id,
        "date": date,
        "daily_total_eur": round(float(result["daily_total_eur"].iloc[0]), 4),
        "risk_p10_eur": round(float(result["risk_p10_eur"].iloc[0]), 4),
        "activated_intervals": int((result["activate"] == 1).sum()),
        "intervals": result.to_dict(orient="records"),
    }


if __name__ == "__main__":
    import uvicorn
    import socket

    def find_free_port(start_port: int = 8000) -> int:
        port = start_port
        while port < 9000:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    port += 1
        raise RuntimeError("No free port found between 8000 and 9000")

    port = find_free_port()
    print(f"Starting server on http://127.0.0.1:{port}")
    print(f"Interactive docs: http://127.0.0.1:{port}/docs")
    uvicorn.run("src.api:app", host="127.0.0.1", port=port, reload=True)
