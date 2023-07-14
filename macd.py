
import pandas_datareader as pdr
import datetime
from matplotlib import pyplot as plt
import numpy as np

#define backtest range
START = datetime.datetime(2005, 1, 1)
END = datetime.datetime(2015, 1, 1)
YEARS = (END - START).days / 365.25

print(YEARS)

#define starting variables
STARTING_BALANCE = 10000

#download data
price = pdr.get_data_yahoo('^GSPC', START, END)

#remove columns
price = price.drop(['High', 'Low', 'Volume', 'Adj Close'], 1)

price.tail()

#calculate benchmark return and balance
price['Return'] = price.Close / price.Open
price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()

price.tail()

#calculate drawdown
price['Bench_Peak'] = price.Bench_Bal.cummax()
price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak

bench_dd = round((price.Bench_DD / price.Bench_Peak).min() * 100, 2)

#calculate MACD
exp1 = price.Close.ewm(span = 12, adjust=False).mean()
exp2 = price.Close.ewm(span = 26, adjust=False).mean()

macd = exp1 - exp2
signal = macd.ewm(span = 9, adjust=False).mean()

plt.figure()

#create subplot for price data
plt.subplot(211)
plt.plot(price.Close)

#create subplot for MACD
plt.subplot(212)
plt.plot(macd)
plt.plot(signal)

#show plot
plt.show()

#add column for entries
price['Long'] = macd > signal

price.tail()

#calculate system return and balance
price['Sys_Ret'] = np.where(price.Long.shift(1) == True, price.Return, 1)

price['Sys_Bal'] = STARTING_BALANCE * price.Sys_Ret.cumprod()

price.tail()

#calculate system drawdown
price['Sys_Peak'] = price.Sys_Bal.cummax()
price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak

sys_dd = round((price.Sys_DD / price.Sys_Peak).min() * 100, 2)

price.tail()

#calculate metrics
bench_return = round(((price.Bench_Bal[-1]/price.Bench_Bal[0]) - 1) * 100, 2)
bench_cagr = round(((((price.Bench_Bal[-1]/price.Bench_Bal[0])**(1/YEARS))-1)*100), 2)
sys_ret = round(((price.Sys_Bal[-1]/price.Sys_Bal[0]) - 1) * 100, 2)
sys_cagr = round(((((price.Sys_Bal[-1]/price.Sys_Bal[0])**(1/YEARS))-1)*100), 2)

print(f'Benchmark Total return: {bench_return}%')
print(f'Benchmark CAGR: {bench_cagr}')
print(f'Benchmark DD: {bench_dd}%')
print('')
print(f'System Total return: {sys_ret}%')
print(f'System CAGR: {sys_cagr}')
print(f'System DD: {sys_dd}%')

plt.plot(price.Bench_Bal)
plt.plot(price.Sys_Bal)

plt.show()


