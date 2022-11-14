from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('RELIANCE.csv')
df.index=pd.to_datetime(df['Date'])
df=df.drop(['Date'], axis='columns')
df['Open-Close']=df.Open - df.Close
df['High-Low']=df.High - df.Low
X=df[['Open-Close', 'High-Low']]
y=np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
split_percentage=0.8
split=int(split_percentage*len(df))
X_train=X[:split]
y_train=y[:split]
X_test=X[split:]
y_test=y[split:]
cls=SVC().fit(X_train, y_train)
df['Predicted_Signal']=cls.predict(X)
df['Return']=df.Close.pct_change()
df['Strategy_Return']=df.Return *df.Predicted_Signal.shift(1)
df['Cumulative_Returns']=df['Return'].cumsum()
df['Cumulative_Strategy']=df['Strategy_Return'].cumsum()
plt.plot(df['Cumulative_Returns'], color='red')
plt.plot(df['Cumulative_Strategy'],color='blue')