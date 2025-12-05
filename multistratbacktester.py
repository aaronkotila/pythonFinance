# MULTI-STRATEGY BACKTESTER
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

print("Downloading SPY...")
data = yf.download("SPY", start="2010-01-01", progress=False, auto_adjust=True)

# CLEAN DATAFRAME AND PRICE COLUMN
price = pd.DataFrame(data['Close'])          # Force it into a clean DataFrame
price.columns = ['price']                    # Rename column to 'price'

# 1. 50/200 SMA
price['sma50']  = price['price'].rolling(50).mean()
price['sma200'] = price['price'].rolling(200).mean()
price['sma_signal'] = (price['sma50'] > price['sma200']).astype(int)

# 2. RSI
delta = price['price'].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.ewm(com=13, adjust=False).mean()
roll_down = down.ewm(com=13, adjust=False).mean()
rs = roll_up / roll_down
price['rsi'] = 100 - (100 / (1 + rs))
price['rsi_signal'] = 0
price.loc[price['rsi'] < 30, 'rsi_signal'] =  1
price.loc[price['rsi'] > 70, 'rsi_signal'] = -1

# 3. MACD
exp12 = price['price'].ewm(span=12, adjust=False).mean()
exp26 = price['price'].ewm(span=26, adjust=False).mean()
price['macd'] = exp12 - exp26
price['macd_sig'] = price['macd'].ewm(span=9, adjust=False).mean()
price['macd_signal'] = np.where(price['macd'] > price['macd_sig'], 1, -1)

# 4. Bollinger Bands
price['bb_mid']   = price['price'].rolling(20).mean()
price['bb_std']   = price['price'].rolling(20).std()
price['bb_upper'] = price['bb_mid'] + 2 * price['bb_std']
price['bb_lower'] = price['bb_mid'] - 2 * price['bb_std']

price['bb_signal'] = 0
price.loc[price['price'] < price['bb_lower'], 'bb_signal'] =  1
price.loc[price['price'] > price['bb_upper'], 'bb_signal'] = -1

# Returns
price['ret'] = price['price'].pct_change()

# Strategy returns
strategies = {
    "50/200 SMA":           price['ret'] * price['sma_signal'].shift(1),
    "RSI Extreme":          price['ret'] * price['rsi_signal'].shift(1),
    "MACD":                 price['ret'] * price['macd_signal'].shift(1),
    "Bollinger Bands":      price['ret'] * price['bb_signal'].shift(1),
    "Buy & Hold":           price['ret']
}

# Plot
plt.figure(figsize=(16,10))
for name, ret in strategies.items():
    equity = (1 + ret.fillna(0)).cumprod()
    equity.plot(label=name, linewidth=2.8)

plt.title("MULTI-STRATEGY SHOWDOWN — SPY 2010–2025", fontsize=22, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Growth of $1")
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

print("\nFINAL PERFORMANCE")
for name, ret in strategies.items():
    total = (1 + ret.fillna(0)).cumprod().iloc[-1] - 1
    print(f"{name:30} → {total:8.1%}")