#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:41:57 2020

@author: Ayca
"""
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import keras #conda install -c conda-forge keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import tensorflow as tf


#conda install keras
df = pd.read_csv('data_mobil_ayca.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'])
df = df.sort_values('DATETIME')
df['TS'] = df.DATETIME.values.astype(np.int64) // 10 ** 9
df['TOTAL'] = df.GELEN + df.GIDEN

#res = df.DATETIME.resample('3T', label='right').sum()
ids = df.NE.unique()
tmp_vendor = df.VENDOR.unique()
print(df.head())
#%%
ids = df.NE.unique()
x = df.groupby('NE')
x = dict(list(x))
dfs = []
for i in ids:
    if len(x[i]) >=1000:
        dfs.append(x[i])

#%%
print("Min",df['DATETIME'].min()) 
print("Min",df['DATETIME'].max()) 
maxlendf = 0
maxlendf_index = 0
for i in range(0,len(dfs)):
    if len(dfs[i]) > maxlendf:
        maxlendf = len(dfs[i])
        maxlendf_index = i
print("Max len: {} of port:{}, index: {}".format( maxlendf, dfs[maxlendf_index].NE, maxlendf_index))
# len = 1302, dfs[685]
inspected = dfs[maxlendf_index]
flag_3G = False
flag_4G = False
for i in range(0, len(inspected)):
    if '3G' in inspected.iloc[i].VENDOR:
        flag_3G = True
    elif '4G' in inspected.iloc[i].VENDOR:
        flag_4G = True
        
plt.plot(dfs[0].DATETIME.values,dfs[0].GELEN.values,'r',dfs[0].DATETIME.values,dfs[0].GIDEN.values,'m',inspected.DATETIME.values,inspected.GELEN.values,'b',inspected.DATETIME,inspected.GIDEN.values,'c')
plt.xlabel("index")
plt.ylabel("MB")
plt.title("{} vs {}".format(dfs[0].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

plt.plot(dfs[0].DATETIME.values,dfs[0].GELEN.values,'b',inspected.DATETIME.values,inspected.GELEN.values,'m')
plt.xlabel("index")
plt.ylabel("GELEN MB")
plt.title("GELEN of {} vs {}".format(dfs[0].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

plt.plot(dfs[0].DATETIME.values,dfs[0].GIDEN.values,'m',inspected.DATETIME,inspected.GIDEN.values,'c')
plt.xlabel("index")
plt.ylabel("GIDEN MB")
plt.title("GIDEN of {} vs {}".format(dfs[0].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

plt.plot(dfs[687].DATETIME.values,dfs[687].GELEN.values,'r',dfs[687].DATETIME.values,dfs[687].GIDEN.values,'m',inspected.DATETIME.values,inspected.GELEN.values,'b',inspected.DATETIME,inspected.GIDEN.values,'c')
plt.xlabel("index")
plt.ylabel("MB")
plt.title("{} vs {}".format(dfs[687].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

plt.plot(dfs[687].DATETIME.values,dfs[687].GELEN.values,'b',inspected.DATETIME.values,inspected.GELEN.values,'m')
plt.xlabel("index")
plt.ylabel("GELEN MB")
plt.title("GELEN of {} vs {}".format(dfs[687].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

plt.plot(dfs[687].DATETIME.values,dfs[687].GIDEN.values,'m',inspected.DATETIME,inspected.GIDEN.values,'c')
plt.xlabel("index")
plt.ylabel("GIDEN MB")
plt.title("GIDEN of {} vs {}".format(dfs[687].NE.iloc[0],inspected.NE.iloc[0]))
plt.show()

#%%
def reshape_set(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
    	# check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    	# gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
    
#%%
def mean_squared_error(Y_true,Y_pred):
    Y_pred = Y_pred[0][0]
    return np.square(np.subtract(Y_true,Y_pred)).mean()
#%%
print("\n20")
gelen_set = inspected['GELEN'].values
giden_set = inspected['GIDEN'].values
end_i = 1300
start_i = 1280
gelen_train = gelen_set[:start_i]
gelen_test = gelen_set[start_i:end_i]
Y_true = gelen_set[end_i] 
n_steps = 20
gelen_x,gelen_y = reshape_set(gelen_train,n_steps)
print("After first split-gelen_x.shape", gelen_x.shape)
# summarize the data
# for i in range(len(gelen_x)):
# 	print(gelen_x[i], gelen_y[i])
#%% Vanilla LSTM

# define model
n_features = 1

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# reshape from [samples, timesteps] into [samples, timesteps, features]
gelen_x = gelen_x.reshape((gelen_x.shape[0], gelen_x.shape[1], n_features))


# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Vanilla LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))
#%% Stacked
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Stacked LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))

#%% CNN
# from keras.layers import Flatten
# from keras.layers import TimeDistributed
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# gelen_x,gelen_y = split_set(gelen_train,n_steps)
# print("CNN-gelen_x.shape", gelen_x.shape)

# # choose a number of time steps
# n_steps = 4

# # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
# n_features = 1
# n_seq = 2
# n_steps = 10
# gelen_x = gelen_x.reshape((gelen_x.shape[0], n_seq, n_steps, n_features))

# # define model
# model = Sequential()
# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(50, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# # fit model
# model.fit(gelen_x, gelen_y, epochs=500, verbose=0)

# #prediction
# gelen_test = gelen_test.reshape((1, n_steps, n_features))
# yhat = model.predict(gelen_test, verbose=0)
# print("Prediction with Stacked LSTM:",yhat)
# print("Real Value:",Y_true)
# print("MSE:",mean_squared_error(Y_true, yhat))
#%%
print("10")
gelen_set = inspected['GELEN'].values
giden_set = inspected['GIDEN'].values
end_i = 1300
start_i = 1290
gelen_train = gelen_set[:start_i]
gelen_test = gelen_set[start_i:end_i]

giden_train = giden_set[:start_i]
giden_test = giden_set[start_i:end_i]

Y_true = gelen_set[end_i] 
Y_true_giden = giden_set[end_i] 
n_steps = 10
gelen_x,gelen_y = reshape_set(gelen_train,n_steps)
giden_x, giden_y = reshape_set(giden_train,n_steps)
print("After first split-gelen_x.shape", gelen_x.shape)
# summarize the data
# for i in range(len(gelen_x)):
# 	print(gelen_x[i], gelen_y[i])
#%% Vanilla LSTM

# define model
n_features = 1

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# reshape from [samples, timesteps] into [samples, timesteps, features]
gelen_x = gelen_x.reshape((gelen_x.shape[0], gelen_x.shape[1], n_features))


# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Vanilla LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))
#%% Stacked
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Stacked LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))

#%%
print("1")
gelen_set = inspected['GELEN'].values
giden_set = inspected['GIDEN'].values
end_i = 1300
start_i = 1299
gelen_train = gelen_set[:start_i]
gelen_test = gelen_set[start_i:end_i]
Y_true = gelen_set[end_i] 
n_steps = 1
gelen_x,gelen_y = reshape_set(gelen_train,n_steps)
print("After first split-gelen_x.shape", gelen_x.shape)
# summarize the data
# for i in range(len(gelen_x)):
# 	print(gelen_x[i], gelen_y[i])
#%% Vanilla LSTM

# define model
n_features = 1

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# reshape from [samples, timesteps] into [samples, timesteps, features]
gelen_x = gelen_x.reshape((gelen_x.shape[0], gelen_x.shape[1], n_features))


# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Vanilla LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))
#%% Stacked
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(gelen_x, gelen_y, epochs=200, verbose=0)

#prediction
gelen_test = gelen_test.reshape((1, n_steps, n_features))
yhat = model.predict(gelen_test, verbose=0)
print("Prediction with Stacked LSTM:",yhat)
print("Real Value:",Y_true)
print("MSE:",mean_squared_error(Y_true, yhat))

#%% Vanilla LSTM method
def Vanilla_meth(seq_x,seq_y,seq_test):
    # define model
    n_features = 1
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    seq_x = seq_x.reshape((seq_x.shape[0], seq_x.shape[1], n_features))
    
    
    # fit model
    model.fit(seq_x, seq_y, epochs=200, verbose=0)
    
    #prediction
    seq_test = seq_test.reshape((1, n_steps, n_features))
    yhat = model.predict(seq_test, verbose=0)
    print("Prediction with Vanilla LSTM:",yhat)
    print("Real Value:",Y_true)
    print("MSE:",mean_squared_error(Y_true, yhat))
    
def Stacked_meth(seq_x,seq_y,seq_test):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(seq_x, seq_y, epochs=200, verbose=0)
    
    #prediction
    seq_test = seq_test.reshape((1, n_steps, n_features))
    yhat = model.predict(seq_test, verbose=0)
    print("Prediction with Stacked LSTM:",yhat)
    print("Real Value:",Y_true)
    print("MSE:",mean_squared_error(Y_true, yhat))