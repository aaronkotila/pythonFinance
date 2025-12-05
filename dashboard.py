# dashboard.py — STREAMLIT
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Aaron's Quant Lab", layout="wide")
st.title("Multi-Strategy Backtester")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()
start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))

if st.sidebar.button("RUN SHOWDOWN"):
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        p = pd.DataFrame(data['Close'])
        p.columns = ['price']

        # 1. 50/200 SMA
        p['sma50']  = p['price'].rolling(50).mean()
        p['sma200'] = p['price'].rolling(200).mean()
        p['sma_sig'] = (p['sma50'] > p['sma200']).astype(int)

        # 2. RSI Extreme
        delta = p['price'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
        rsi = 100 - (100 / (1 + rs))
        p['rsi_sig'] = 0
        p.loc[rsi < 30, 'rsi_sig'] =  1
        p.loc[rsi > 70, 'rsi_sig'] = -1

        # 3. MACD
        exp12 = p['price'].ewm(span=12, adjust=False).mean()
        exp26 = p['price'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        p['macd_sig'] = np.where(macd > macd_sig, 1, -1)

        # 4. Bollinger Bands
        p['bb_mid'] = p['price'].rolling(20).mean()
        p['bb_std'] = p['price'].rolling(20).std()
        p['bb_upper'] = p['bb_mid'] + 2*p['bb_std']
        p['bb_lower'] = p['bb_mid'] - 2*p['bb_std']
        p['bb_sig'] = 0
        p.loc[p['price'] < p['bb_lower'], 'bb_sig'] =  1
        p.loc[p['price'] > p['bb_upper'], 'bb_sig'] = -1

        # Returns
        p['ret'] = p['price'].pct_change()

        # Plot ALL strategies
        fig, ax = plt.subplots(figsize=(15, 8))
        (1 + p['ret'] * p['sma_sig'].shift(1).fillna(0)).cumprod().plot(ax=ax, label="50/200 SMA")
        (1 + p['ret'] * p['rsi_sig'].shift(1).fillna(0)).cumprod().plot(ax=ax, label="RSI Extreme")
        (1 + p['ret'] * p['macd_sig'].shift(1).fillna(0)).cumprod().plot(ax=ax, label="MACD")
        (1 + p['ret'] * p['bb_sig'].shift(1).fillna(0)).cumprod().plot(ax=ax, label="Bollinger Bands")
        (1 + p['ret'].fillna(0)).cumprod().plot(ax=ax, label="Buy & Hold", alpha=0.6, linewidth=3)

        ax.set_title(f"{ticker} — MULTI-STRATEGY SHOWDOWN", fontsize=20, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.success("All 4 strategies + Buy & Hold loaded!")