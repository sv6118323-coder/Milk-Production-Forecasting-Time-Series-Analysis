# 🥛 Monthly Milk Production Forecasting

A complete time series forecasting project using classical statistical models (ARIMA, SARIMA) and deep learning models (Simple RNN, LSTM) on the Monthly Milk Production dataset (1962–1975).

---

## 📁 Project Structure

```
milk-forecasting/
├── data/
│   └── monthly_milk_production.csv
├── timeseries_milk_forecasting.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Source:** Monthly Milk Production dataset
- **Period:** January 1962 – December 1975 (168 months)
- **Target:** Milk production in pounds per cow per month
- **Train/Test Split:** 156 months train / 12 months test

---

## 🔬 Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| ARIMA | Statistical | AutoRegressive Integrated Moving Average — no seasonal component |
| SARIMA Baseline | Statistical | Seasonal ARIMA with fixed order (1,1,1)(1,1,1,12) |
| Simple RNN | Deep Learning | Recurrent Neural Network with 12-step input window |
| LSTM | Deep Learning | Long Short-Term Memory network with 100 units |
| SARIMA Tuned | Statistical | Grid-search optimized SARIMA selected via AIC criterion |

---

## 🏆 Final Results

| Model | MAE | RMSE | MAPE | R² | Rank |
|-------|-----|------|------|----|------|
| 🥇 **SARIMA Tuned** | **9.45** | **10.61** | **1.09%** | **0.9616** | 1st |
| 🥈 SARIMA Baseline | 9.74 | 10.70 | 1.13% | 0.9610 | 2nd |
| 🥉 Simple RNN | 16.72 | 21.15 | 1.95% | 0.8475 | 3rd |
| LSTM | 41.61 | 48.11 | 4.79% | 0.2111 | 4th |
| ARIMA | 59.59 | 75.52 | 6.62% | −0.9438 | 5th |

### 🔍 What the Metrics Mean

| Metric | Meaning | Goal |
|--------|---------|------|
| **MAE** | Average absolute error in lbs/cow | Lower is better |
| **RMSE** | Penalizes large errors more heavily | Lower is better |
| **MAPE** | Average % error per prediction | Lower is better |
| **R²** | % of variance explained (1.0 = perfect) | Higher is better |

---

## 📌 Key Findings

**1. SARIMA Tuned is the best model (R² = 0.9616)**
Grid search over all `(p,d,q)(P,D,Q,12)` combinations using AIC improved upon the baseline SARIMA — reducing MAE from 9.74 → 9.45 and pushing R² from 0.9610 → 0.9616. At 1.09% MAPE, predictions are within ~1% of actual values on average.

**2. ARIMA completely fails (R² = −0.9438)**
A negative R² means ARIMA performs *worse than simply predicting the mean*. Without a seasonal component, it outputs a smooth trend line while the actual data oscillates sharply every 12 months — all those missed peaks and troughs collapse the score.

**3. LSTM underperforms despite its reputation (R² = 0.2111)**
With only 156 training samples, LSTM doesn't have enough data to learn long-range dependencies. Rolling forecast error accumulation over 12 steps compounds the problem further. Deep learning is not always the best choice.

**4. Classical beats Deep Learning here**
SARIMA encodes seasonality *directly* through its parameters. LSTM has to *discover* the 12-month pattern from scratch — with a small dataset, it never gets there. SARIMA wins by a large margin.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/milk-forecasting.git
cd milk-forecasting
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
python timeseries_milk_forecasting.py
```

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key libraries:

- `pandas`, `numpy` — data handling and numerical operations
- `statsmodels` — ARIMA, SARIMA modelling
- `tensorflow` / `keras` — RNN and LSTM architectures
- `scikit-learn` — MinMaxScaler, MAE, RMSE, R² evaluation
- `matplotlib` — all visualizations

---

## 📈 Concepts Covered

- Time series EDA — trend, seasonality, noise
- ACF / PACF analysis for parameter selection
- Seasonal decomposition
- Stationarity and differencing
- MinMaxScaler + TimeseriesGenerator for deep learning
- Rolling forecast strategy for multi-step prediction
- SARIMA hyperparameter tuning via grid search (AIC)
- Evaluation using MAE, RMSE, MAPE and R²

---

## 💡 Key Takeaway

> For **small, highly seasonal time series**, classical statistical models (SARIMA) decisively outperform deep learning (LSTM).
> Deep learning only gains an edge with **larger datasets (1000+ samples)** or **complex multivariate signals** where seasonality cannot be cleanly specified.
