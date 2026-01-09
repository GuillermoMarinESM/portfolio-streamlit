import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Forecasting Models Comparison")

st.write("""
Comparison of baseline forecasting models on a time series.
Focus: understanding model behavior and error, not complexity.
""")

# --- Generate synthetic time series ---
np.random.seed(42)
n = 200
trend = np.linspace(0, 10, n)
noise = np.random.normal(0, 1, n)
series = trend + noise

df = pd.DataFrame({
    "time": range(n),
    "value": series
})

# --- User controls ---
train_size = st.slider("Training size (%)", 50, 90, 70)
window = st.slider("Moving average window", 2, 10, 5)

split = int(n * train_size / 100)

train = df.iloc[:split]
test = df.iloc[split:]

# --- Models ---
# Naive
naive_forecast = np.repeat(train["value"].iloc[-1], len(test))

# Moving Average
ma_value = train["value"].iloc[-window:].mean()
ma_forecast = np.repeat(ma_value, len(test))

# --- Metrics ---
mae_naive = np.mean(np.abs(test["value"] - naive_forecast))
mae_ma = np.mean(np.abs(test["value"] - ma_forecast))

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(train["time"], train["value"], label="Train")
ax.plot(test["time"], test["value"], label="Test")
ax.plot(test["time"], naive_forecast, "--", label="Naive Forecast")
ax.plot(test["time"], ma_forecast, "--", label="Moving Average Forecast")

ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Value")

st.pyplot(fig)

# --- Results ---
st.subheader("Model Performance (MAE)")

st.metric("Naive Forecast MAE", round(mae_naive, 2))
st.metric("Moving Average MAE", round(mae_ma, 2))

# --- Interpretation ---
st.subheader("Interpretation")

st.write("""
- The naive model serves as a strong baseline.
- The moving average smooths recent noise but may lag under strong trends.
- Lower MAE indicates better short-term forecasting performance.

In practice, these baselines help determine whether more complex models are justified.
""")
