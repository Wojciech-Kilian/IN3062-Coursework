# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:48:52 2021

@author: wkxxx
"""

import pandas as pd
import io
import requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn import preprocessing
import matplotlib.pyplot as plt 

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_
# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

##############################################################################

path = "."

filename = os.path.join(path,"Life Expectancy Data.csv")    
df = pd.read_csv(filename,na_values=['NA','?'])

df.drop('Income composition of resources', 1, inplace=True)
df.drop('Population', 1, inplace=True)
df.drop('Total expenditure', 1, inplace=True)
df.drop('Country', 1, inplace=True)

encode_text_index(df,'Status')

headers = list(df.columns.values)
df = df.dropna()
#Dropping worked out better
#for field in headers:
#    med = df[field].median()
#    df[field] = df[field].fillna(med)
#print(df.isnull().any())
X,y = to_xy(df,"Life expectancy ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=0)


model = Sequential()
model.add(Dense(40, input_dim=X.shape[1], activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))) # Hidden 1
model.add(Dropout(0.05)) #Hidden 2
model.add(Dense(40, activation='sigmoid')) #Hidden 3
model.add(Dense(40, activation='relu')) # Hidden 4
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='loss', patience=5,min_delta=0.01, mode='min')
training_trace=model.fit(X_train,y_train,verbose=2,epochs=100,validation_split=0.25,callbacks=[monitor])

model.summary()

#model.save(os.path.join(".","network.h5"))

pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Final score (RMSE): {score}")
for i in range(10):
    print(f"{i+1}. Life expectancy: {y[i]}, predicted Life expectancy: {pred[i]}")
    
plt.figure(figsize=(10,10))
plt.plot(training_trace.history['loss'])
plt.plot(training_trace.history['val_loss'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.figure(figsize=(10,10))
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()

chart_regression(pred[:50].flatten(),y_test[:50],sort=True)    
chart_regression(pred[:100].flatten(),y_test[:100],sort=True)