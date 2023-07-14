
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aapl_df = yf.download("AAPL", start="2018-03-24", end="2023-03-24")


aapl_df['ema_short'] = aapl_df['Close'].ewm(span=20, adjust=False).mean()
aapl_df['ema_long'] = aapl_df['Close'].ewm(span=50, adjust=False).mean()


aapl_df['bullish'] = 0.0
aapl_df['bullish'] = np.where(aapl_df['ema_short'] > aapl_df['ema_long'], 1.0, 0.0)
aapl_df['crossover'] = aapl_df['bullish'].diff()


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111, ylabel='Price in $')

aapl_df['Close'].plot(ax=ax1, color='b', lw=2.)
aapl_df['ema_short'].plot(ax=ax1, color='r', lw=2.)
aapl_df['ema_long'].plot(ax=ax1, color='g', lw=2.)

ax1.plot(aapl_df.loc[aapl_df.crossover == 1.0].index, 
         aapl_df.Close[aapl_df.crossover == 1.0],
         '^', markersize=10, color='g')
ax1.plot(aapl_df.loc[aapl_df.crossover == -1.0].index, 
         aapl_df.Close[aapl_df.crossover == -1.0],
         'v', markersize=10, color='r')
plt.legend(['Close', 'EMA Short', 'EMA Long', 'Buy', 'Sell'])
plt.title('AAPL EMA Crossover')