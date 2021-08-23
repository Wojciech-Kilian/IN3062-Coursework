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
from sklearn.model_selection import KFold

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
df.drop('Country', 1, inplace=True)
encode_text_index(df,'Status')

headers = list(df.columns.values)

for field in headers:
    med = df[field].median()
    df[field] = df[field].fillna(med)
#print(df.isnull().any())
X,y = to_xy(df,"Life expectancy ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=0)

model = Sequential()
model.add(Dense(40, input_dim=X.shape[1], activation='sigmoid')) # Hidden 1
model.add(Dropout(0.04))
model.add(Dense(40, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
#model.summary() #note, only works if input shape specified, or Input layer given
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=100)
model.summary()
pred = model.predict(X_test)
print("Shape: {}".format(pred.shape))
print(pred[:10])
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"Final score (RMSE): {score}")
for i in range(10):
    print(f"{i+1}. Life expectancy: {y[i]}, predicted Life expectancy: {pred[i]}")