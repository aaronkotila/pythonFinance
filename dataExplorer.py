import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import seaborn as sns
import matplotlib.pyplot as plt

st.title("--Financial Data Explorer--")
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Ticker Symbol", "MU")
start_date = st.sidebar.date_input("Start Date", dt.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2025, 12, 31))

@st.cache_data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, axis=1, level=1)
    return data

st.write(f"Showing data for **${ticker}**")
try:
    df = get_data(ticker, start_date, end_date)
    df['Returns'] = df['Close'].pct_change()
    
    last_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    daily_change = last_price - prev_price

    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

    volatility = df['Returns'].std() * (252 ** 0.5)

    st.markdown("### Key Metrics")
    col1, col2, col3, = st.columns(3)
    col1.metric(
        label="Last Price",
        value=f"${last_price:.2f}",
        delta=f"{daily_change:.2f}"
    )
    col2.metric(
        label="Total Return (Period)",
        value=f"{total_return:.2%}",
        delta_color="off"
    )
    col3.metric(
        label="Annualized Volatility",
        value=f"{volatility:.2%}",
        help="Standard Deviation * Sqrt(252). Higher = Riskier."
    )

    st.divider()

    df.dropna(inplace=True)

    st.subheader(f"${ticker} Price Action (1D vs 50SMA)")

    fig, ax = mpf.plot(df, type='candle', style='yahoo', mav=(50), volume=True, returnfig=True)

    st.pyplot(fig)

    st.subheader("Risk Profile (Volatility)")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.histplot(df['Returns'], bins=50, kde=True, color='blue', ax=ax2)
    ax2.set_title("Distribution of Daily Returns")

    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")