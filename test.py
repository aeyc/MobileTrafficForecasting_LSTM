#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:41:57 2020

@author: Ayca
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

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

#plt.plot()
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
        
#type(inspected.iloc[0].GELEN)
#Out[17]: numpy.float64 


#%%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import tensorflow as tf

#%%
def shape_func(data):
    data = data.values
    tstemps,features = data.shape
    data = data.reshape((tstemps,1,features))
    return data
print("Describe:\n",inspected.describe(),"\n")
#%%

inspected_features = pd.concat([inspected.TS,inspected.VENDOR,inspected.NE],axis = 1)
#inspected_features = inspected.TS
#inspected_result = pd.concat([inspected.GELEN,inspected.GIDEN],axis = 1)
inspected_result = inspected.GELEN


#dataset = inspected.values #array
#train_split = dataset[0:1000]
#test_split = dataset[1000:]
#test_split = dataset[1000:]
#x_train = pd.concat([train_split.VENDOR,trainsplit.NE])


train_x = inspected_features.iloc[0:1000]
train_y = inspected_result.iloc[0:1000]

print("train_x.shape:",train_x.shape)
print("train_y.shape:",train_y.shape)

#train_x = train_x.values
#print("type(train_x)",type(train_x))
#trainx_timestemps,trainx_features = train_x.shape
#train_x = train_x.reshape((1,trainx_timestemps,trainx_features))

train_x = shape_func(train_x) #for 3 feature pred
#train_x = train_x.values
train_y = train_y.values

test_x = inspected_features.iloc[1000:]
test_y = inspected_result.iloc[1000:]
print("test_x.shape:",test_x.shape)
print("test_y.shape:",test_y.shape)

test_x = shape_func(test_x) #for 3 feature pred
#test_x = test_x.values
test_y = test_y.values

model = Sequential()
#input_shape = timestamps,features
#TRIED: input_shape=(1000,2); 3; train_x.shape;(1000,1),(None,3),(1000,3)


model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

"""
One Epoch is when an ENTIRE dataset is passed forward 
and backward through the neural network only ONCE.
"""
#model.fit(train_x,
#          train_y,
#          epochs=3,
#          validation_data=(test_x, test_y))

model.fit(train_x,
          train_y)