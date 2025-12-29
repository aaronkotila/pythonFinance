import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# APP CONFIGURATION
st.set_page_config(page_title="Quant Pairs Trader", layout="wide")
st.title("Statistical Arbitrage: Pairs Trading Dashboard")

# Sidebar for controls
st.sidebar.header("Strategy Settings")

# Pair Selection
pairs = {
    "Coke vs Pepsi (Consumer Staples)": ("KO", "PEP"),
    "Gold vs Silver (Precious Metals)": ("GLD", "SLV"),
    "Chevron vs Exxon (Energy)": ("CVX", "XOM"),
    "Nvidia vs AMD (Semiconductors)": ("NVDA", "AMD"),
    "Google vs Microsoft (Tech Giants)": ("GOOG", "MSFT")
}

pair_name = st.sidebar.selectbox("Select Asset Pair:", list(pairs.keys()))
ticker_a, ticker_b = pairs[pair_name]

# Lookback Window Slider
window = st.sidebar.slider("Z-Score Lookback (Days):", min_value=10, max_value=100, value=30)
entry_threshold = st.sidebar.number_input("Entry Threshold (Std Dev):", value=2.0, step=0.1)

# DATA LOADING (Cached)
@st.cache_data
def get_data(t1, t2):
    # auto_adjust=False to fix the FutureWarning and get raw Close
    df = yf.download([t1, t2], period="2y", progress=False, auto_adjust=False)['Close']
    
    # Drop NaNs just in case
    df = df.dropna()
    
    # Rename columns explicitly to avoid confusion
    df = df[[t1, t2]].copy() # Enforce order
    return df

try:
    data = get_data(ticker_a, ticker_b)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# CALCULATIONS
# Calculate Ratio
data['Ratio'] = data[ticker_a] / data[ticker_b]

# Calculate Z-Score
rolling_mean = data['Ratio'].rolling(window=window).mean()
rolling_std = data['Ratio'].rolling(window=window).std()
data['Z_Score'] = (data['Ratio'] - rolling_mean) / rolling_std

# Calculate Daily Returns for individual assets
data['Ret_A'] = data[ticker_a].pct_change()
data['Ret_B'] = data[ticker_b].pct_change()

# BACKTEST LOGIC (Event-Driven Loop)
# Logic: Enter at +/- Threshold. Exit when Z-Score crosses 0 (Mean Reversion).
positions = []
current_position = 0 # 1 = Long Ratio, -1 = Short Ratio, 0 = Flat

for z in data['Z_Score']:
    if current_position == 0:
        # ENTRY CONDITIONS
        if z < -entry_threshold:
            current_position = 1 # Long A, Short B
        elif z > entry_threshold:
            current_position = -1 # Short A, Long B
    else:
        # EXIT CONDITION (Crosses Mean)
        # If we are Long (1) and Z goes positive, or Short (-1) and Z goes negative
        if (current_position == 1 and z >= 0) or (current_position == -1 and z <= 0):
            current_position = 0
            
    positions.append(current_position)

data['Position'] = positions

# Strategy Return:
# If Long Ratio (Pos=1): Profit if A up, B down -> (Ret_A - Ret_B)
# If Short Ratio (Pos=-1): Profit if B up, A down -> (Ret_B - Ret_A) or -1 * (Ret_A - Ret_B)
# We shift position by 1 because we make the decision based on Today's Z-Score, 
# but realize the return Tomorrow.
data['Spread_Ret'] = data['Ret_A'] - data['Ret_B']
data['Strategy_Ret'] = data['Position'].shift(1) * data['Spread_Ret']

# Cumulative PnL
data['Cumulative_PnL'] = (1 + data['Strategy_Ret']).cumprod()

# DASHBOARD VISUALS

# KPI Metrics
col1, col2, col3 = st.columns(3)
latest_z = data['Z_Score'].iloc[-1]
total_return = (data['Cumulative_PnL'].iloc[-1] - 1) * 100

col1.metric("Latest Z-Score", f"{latest_z:.2f}")
col2.metric("Total Return (2Y)", f"{total_return:.2f}%")
col3.metric("Current Position", "LONG" if positions[-1]==1 else "SHORT" if positions[-1]==-1 else "FLAT")

# PLOT 1: Z-Score & Signals
st.subheader(f"1. Mean Reversion Signal ({ticker_a}/{ticker_b})")
fig_z = plt.figure(figsize=(12, 4))
plt.plot(data.index, data['Z_Score'], label="Z-Score", color='black', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(entry_threshold, color='red', linestyle='--', label="Short Threshold")
plt.axhline(-entry_threshold, color='green', linestyle='--', label="Long Threshold")

# Fill zones
plt.fill_between(data.index, data['Z_Score'], entry_threshold, where=(data['Z_Score'] > entry_threshold), color='red', alpha=0.3)
plt.fill_between(data.index, data['Z_Score'], -entry_threshold, where=(data['Z_Score'] < -entry_threshold), color='green', alpha=0.3)
plt.legend(loc='upper left')
plt.ylabel("Std Dev")
st.pyplot(fig_z)

# PLOT 2: Cumulative PnL
st.subheader("2. Hypothetical Strategy Performance")
fig_pnl = plt.figure(figsize=(12, 4))
plt.plot(data.index, data['Cumulative_PnL'], label="Strategy Equity Curve", color='#1f77b4', linewidth=2)
plt.axhline(1.0, color='black', linestyle='-', alpha=0.3)
plt.ylabel("Growth of $1")
plt.legend(loc='upper left')
st.pyplot(fig_pnl)

# PLOT 3: Raw Prices
st.subheader(f"3. Underlying Prices ({ticker_a} vs {ticker_b})")
fig_price = plt.figure(figsize=(12, 4))
# Normalize to start at 100 for comparison
norm_a = (data[ticker_a] / data[ticker_a].iloc[0]) * 100
norm_b = (data[ticker_b] / data[ticker_b].iloc[0]) * 100
plt.plot(data.index, norm_a, label=f"{ticker_a} (Normalized)", color='red')
plt.plot(data.index, norm_b, label=f"{ticker_b} (Normalized)", color='blue')
plt.ylabel("Normalized Price")
plt.legend()
st.pyplot(fig_price)

# Show Data Table
with st.expander("See Raw Data"):

    st.dataframe(data.tail(20))

