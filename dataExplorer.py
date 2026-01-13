import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Explorer", layout="wide")

st.title("--Financial Data Explorer--")
st.sidebar.header("User Input")

ticker = st.sidebar.text_input("Ticker Symbol", "TSLA").upper()
start_date = st.sidebar.date_input("Start Date", dt.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2025, 12, 31))

if ticker:
    try:
        stock_info = yf.Ticker(ticker).info
        company_name = stock_info.get('shortName', ticker)
        st.sidebar.success(f"Valid Ticker: {company_name}")
    except:
        st.sidebar.warning("Ticker metadata not found. Double check symbol.")

@st.cache_data
def get_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            
            if symbol in data.columns.get_level_values(1):
                data = data.xs(symbol, axis=1, level=1)
                
            elif isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
                
        return data
    except Exception as e:
        return pd.DataFrame()

st.write(f"Showing data for **${ticker}**")

try:
    df = get_data(ticker, start_date, end_date)

    if df.empty:
        st.error(f"No data found for **{ticker}** in this date range. Please check the Ticker symbol or expand the dates.")
        st.stop()

    if len(df) < 2:
        st.warning("Not enough data points (need at least 2 days) to calculate returns. Try expanding the date range.")
        st.stop()
        
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

    plot_df = df.dropna()

    st.subheader(f"${ticker} Price Action (1D vs 50SMA)")

    mav_val = 50 if len(plot_df) > 50 else len(plot_df) // 2

    fig, ax = mpf.plot(df, type='candle', style='yahoo', mav=(50), volume=True, returnfig=True)
    st.pyplot(fig)

    st.subheader("Risk Profile (Volatility)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Returns'], bins=50, kde=True, color='blue', ax=ax2)
    ax2.set_title("Distribution of Daily Returns")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")
