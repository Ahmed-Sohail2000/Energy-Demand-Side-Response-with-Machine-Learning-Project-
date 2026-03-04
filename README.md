# Energy Demand Side Response with Machine Learning

A simplified DSR (Demand Side Response) system that ingests energy and market data, identifies high-value activation windows, and estimates potential revenue from flexibility.

Built as a technical case study for Cledion. Live app: https://demandsideresponse.streamlit.app/

---

## Architecture Overview

data/                  # Raw input datasets (load, prices)
src/
  ingestion.py         # Load, clean and resample data to 30-min intervals
  features.py          # Feature engineering (lags, rolling averages, price deltas)
  signal.py            # DSR signal: score (0-1) + binary activation per interval
  activation.py        # Site activation engine
  revenue.py           # Revenue simulation (flexible kW, daily total, P10 risk)
app/
  main.py              # Streamlit dashboard (load + price + signal, top windows, revenue)
tests/
  test_activation.py   # 10 unit tests
  test_revenue.py      # 11 unit tests

---

## How to Run

1. Clone the repo
git clone https://github.com/Ahmed-Sohail2000/Energy-Demand-Side-Response-with-Machine-Learning-Project.git
cd Energy-Demand-Side-Response-with-Machine-Learning-Project

2. Set up the environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

3. Run the Streamlit dashboard
streamlit run app/main.py
Open your browser at http://localhost:8501

---

## Key Assumptions

- 10% flexible load per site
- Max 300 kW reduction per interval
- Revenue = flexible kW x DSR rate x 0.5 hr
- P10 risk metric = worst 10% outcome with small noise on reduction

---

## Tech Stack

| Technology     | Purpose                        |
|----------------|--------------------------------|
| Python 3.11    | Core language                  |
| Streamlit      | Dashboard                      |
| Pandas / NumPy | Data handling                  |
| Scikit-learn   | ML model (Logistic Regression) |
| Plotly         | Visualisations                 |

---

## Author

Ahmed Sohail — https://github.com/Ahmed-Sohail2000
