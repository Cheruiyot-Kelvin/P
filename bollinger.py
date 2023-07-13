
import MetaTrader5 as mt5
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

mt5.initialize()

bars = mt5.copy_rates_range("GBPUSD", mt5.TIMEFRAME_D1, 
                            datetime(2020, 1, 1), datetime.now())

bars

df = pd.DataFrame(bars)
df

df['time'] = pd.to_datetime(df['time'], unit='s')
df

fig = px.line(df, x='time', y='close')
fig

df['sma'] = df['close'].rolling(20).mean()

df['sd'] = df['close'].rolling(20).std()

df['lb'] = df['sma'] - 2 * df['sd']

# calculate upper band
df['ub'] = df['sma'] + 2 * df['sd']

df.dropna(inplace=True)
df

# plotting close prices with bollinger bands
fig = px.line(df, x='time', y=['close', 'sma', 'lb', 'ub'])
fig

# find signals

def find_signal(close, lower_band, upper_band):
    if close < lower_band:
        return 'buy'
    elif close > upper_band:
        return 'sell'
    
    
df['signal'] = np.vectorize(find_signal)(df['close'], df['lb'], df['ub'])

df

# creating backtest and position classes

class Position:
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = 'open'
        
    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        self.profit = (self.close_price - self.open_price) * self.volume if self.order_type == 'buy' \
                                                                        else (self.open_price - self.close_price) * self.volume
        self.status = 'closed'
        
    def _asdict(self):
        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'profit': self.profit,
            'status': self.status,
        }
        
        
class Strategy:
    def __init__(self, df, starting_balance, volume):
        self.starting_balance = starting_balance
        self.volume = volume
        self.positions = []
        self.data = df
        
    def get_positions_df(self):
        df = pd.DataFrame([position._asdict() for position in self.positions])
        df['pnl'] = df['profit'].cumsum() + self.starting_balance
        return df
        
    def add_position(self, position):
        self.positions.append(position)
        
    def trading_allowed(self):
        for pos in self.positions:
            if pos.status == 'open':
                return False
        
        return True
        
    def run(self):
        for i, data in self.data.iterrows():
            
            if data.signal == 'buy' and self.trading_allowed():
                sl = data.close - 3 * data.sd
                tp = data.close + 2 * data.sd
                self.add_position(Position(data.time, data.close, data.signal, self.volume, sl, tp))
                
            elif data.signal == 'sell' and self.trading_allowed():
                sl = data.close + 3 * data.sd
                tp = data.close - 2 * data.sd
                self.add_position(Position(data.time, data.close, data.signal, self.volume, sl, tp))
                
            for pos in self.positions:
                if pos.status == 'open':
                    if (pos.sl >= data.close and pos.order_type == 'buy'):
                        pos.close_position(data.time, pos.sl)
                    elif (pos.sl <= data.close and pos.order_type == 'sell'):
                        pos.close_position(data.time, pos.sl)
                    elif (pos.tp <= data.close and pos.order_type == 'buy'):
                        pos.close_position(data.time, pos.tp)
                    elif (pos.tp >= data.close and pos.order_type == 'sell'):
                        pos.close_position(data.time, pos.tp)
                        
        return self.get_positions_df()



# %%
# run the backtest
bollinger_strategy = Strategy(df, 10000, 100000)
result = bollinger_strategy.run()

result

# %%
# plotting close prices with bollinger bands
fig = px.line(df, x='time', y=['close', 'sma', 'lb', 'ub'])

# adding trades to plots
for i, position in result.iterrows():
    if position.status == 'closed':
        fig.add_shape(type="line",
            x0=position.open_datetime, y0=position.open_price, x1=position.close_datetime, y1=position.close_price,
            line=dict(
                color="green" if position.profit >= 0 else "red",
                width=3)
            )
fig

# %%
# visualizing the results of the backtest
px.line(result, x='close_datetime', y='pnl')

# %%



