# 🥛 Monthly Milk Production Forecasting

> Time Series Forecasting using **ARIMA, SARIMA, RNN & LSTM**  
> Monthly Milk Production Dataset · 1962 – 1975

---

## 📌 Project Overview

This project walks through a **complete time series forecasting pipeline** on the Monthly Milk Production dataset. It progressively moves from classical statistical models (ARIMA, SARIMA) to deep learning models (Simple RNN, LSTM), evaluates each model fairly, and concludes with the best-performing model along with hyperparameter tuning.

**Dataset:** Monthly milk production (pounds per cow) from 1962–1975  
**Goal:** Forecast future milk production based on historical seasonal patterns

---

## 🏆 Model Comparison

| Rank | Model | MAE | RMSE | MAPE | R² Score |
|------|-------|-----|------|------|----------|
| 🥇 1 | **SARIMA Tuned** | ~9.45 | ~10.61 | ~1.09% | **0.9616** |
| 🥈 2 | SARIMA Baseline | 9.74 | 10.70 | 1.13% | 0.9610 |
| 🥉 3 | Simple RNN | 16.72 | 21.15 | 1.95% | 0.8475 |
| 4 | LSTM | 41.61 | 48.11 | 4.79% | 0.2111 |
| 5 | ARIMA | 59.59 | 75.52 | 6.62% | -0.9438 |

> ✅ **SARIMA (Tuned)** is the clear winner — R² of 0.9616 means it explains **96.16% of variance** in actual test values.

---

## 🗂️ Project Structure

```
timeseries_project/
│
├── notebook/
│   └── timeseries_milk_forecasting.ipynb   # Complete analysis notebook
│
├── data/
│   └── monthly_milk_production.csv         # Raw dataset (1962–1975)
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/timeseries-milk-forecasting.git
cd timeseries-milk-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebook/timeseries_milk_forecasting.ipynb
```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
statsmodels
scikit-learn
tensorflow
keras
jupyter
```

---

## 📋 Notebook Sections

### 1. 📦 Imports & Data Loading
- Loads `monthly_milk_production.csv` using pandas
- Sets `Date` as datetime index for time series operations

### 2. 📊 Exploratory Data Analysis (EDA)
- Time series plot — identifies **upward trend** and **yearly seasonality**
- **ACF & PACF plots** — confirms 12-month seasonal cycle, determines AR/MA orders
- **Seasonal Decomposition** — breaks series into Trend + Seasonality + Residual

### 3. ✂️ Train / Test Split
- Sequential split — first **156 months** for training, remaining for testing
- No random shuffle — temporal order is preserved (critical for time series)

### 4. 📏 Evaluation Metrics
All models evaluated on 4 metrics:

| Metric | What it measures |
|--------|-----------------|
| **MAE** | Average error in original units |
| **RMSE** | Penalises large errors more heavily |
| **MAPE** | Error as a percentage |
| **R²** | Proportion of variance explained (closer to 1 = better) |

### 5. 🔷 Model 1 — ARIMA(2,1,2)
- Classical statistical model — handles trend but **no seasonality**
- Result: R² = −0.9438 — worst performer, worse than predicting the mean
- Verdict: ❌ Not suitable for seasonal data

### 6. 🔷 Model 2 — SARIMA(1,1,1)(1,1,1,12)
- Extends ARIMA with a **seasonal component** (12-month cycle)
- Result: R² = 0.9610, MAPE = 1.13%
- Verdict: ✅ Excellent performance — best classical model

### 7. 🔷 Model 3 — Simple RNN
- Deep learning model with sliding window of 12 months
- Data scaled using MinMaxScaler before training
- Result: R² = 0.8475, MAPE = 1.95%
- Verdict: ⚠️ Decent but limited by small dataset size

### 8. 🔷 Model 4 — LSTM
- More powerful deep learning model with gated memory cells
- Result: R² = 0.2111, MAPE = 4.79%
- Verdict: ❌ Overfits on small dataset — rolling forecast errors compound

### 9. 🔧 Hyperparameter Tuning — SARIMA Grid Search
- Grid search over `p ∈ [0,1,2]`, `q ∈ [0,1,2]`, `P ∈ [0,1]`, `Q ∈ [0,1]`
- Uses **AIC (Akaike Information Criterion)** to rank configurations
- Tuned model further improves SARIMA baseline performance

---

## 🔍 Why SARIMA Won

| Model | Key Reason |
|-------|-----------|
| **SARIMA** 🥇 | Explicitly models the 12-month seasonal cycle — perfectly matched to this data |
| **LSTM** ❌ | Too complex for 156 training samples — overfits noise |
| **Simple RNN** 🥉 | Learns patterns reasonably but limited by vanishing gradient |
| **ARIMA** ❌ | No seasonal component — R² < 0 confirms it fails completely |

> 💡 **Key Takeaway:** For **short, highly seasonal time series**, classical SARIMA outperforms deep learning. LSTM only gains an edge with **1000+ samples** or complex multi-variate signals.

---

## 📊 Dataset

- **Source:** Monthly Milk Production dataset (classic time series benchmark)
- **Period:** January 1962 – December 1975
- **Frequency:** Monthly
- **Target:** Milk production in pounds per cow
- **Size:** 168 records total (156 train / 12 test)

---

## 🛠️ Tech Stack

| Purpose | Library |
|---------|---------|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib |
| Statistical models | statsmodels (ARIMA, SARIMA) |
| Deep learning | TensorFlow / Keras (RNN, LSTM) |
| Preprocessing | scikit-learn (MinMaxScaler) |
| Notebook | Jupyter |

---

## 👨‍💻 Author

Time Series Forecasting Project  
Built with ❤️ using Python, statsmodels & TensorFlow
