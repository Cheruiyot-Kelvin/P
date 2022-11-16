import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mandates
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from datetime import datetime as sdt
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
d=pd.read_csv(input("Enter csv file: "), na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
outp=pd.DataFrame(d['Adj Close'])
df=d.copy()
df.drop(['Adj Close','Volume'], axis=1)
features=['Open','High','Low','Volume']
scaler=MinMaxScaler()
feature_transform=scaler.fit_transform(df[features])
feature_transform
feature_transform=pd.DataFrame(columns=features, data=feature_transform, index=d.index)
timesplit=TimeSeriesSplit(n_splits=10)
X=feature_transform
timesplit.get_n_splits
for train_index,test_index in timesplit.split(X):
  x_train, x_test=X[:7444],X[7445:],
  y_train, y_test=outp[:7444].values.ravel(), outp[7445:].values.ravel()
trainx=np.array(x_train)
testx=np.array(x_test)
X_train=trainx.reshape(x_train.shape[0],1,x_train.shape[1])
X_test=testx.reshape(x_test.shape[0],1,x_test.shape[1])
lstm=Sequential()
lstm.add(LSTM(32, input_shape=(1,trainx.shape[1]),activation='relu',return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
plot_model(lstm, show_shapes=True,show_layer_names=True)
hist=lstm.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1, shuffle=False)
y_pred=lstm.predict(X_test)
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title("Pred by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled Usd')
plt.legend()
plt.show()
