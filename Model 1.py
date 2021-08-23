# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:29:20 2021

@author: wkxxx
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt  

path = "."

filename_read = os.path.join(path, "Life Expectancy Data.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

# First data frame
df1 = df.select_dtypes(include=['int', 'float'])

headers = list(df1.columns.values)
fields = []
     
#repeating with dropping values
for field in headers:
    df1 = df1.dropna()
    fields.append({
        'name' : field,
        #'mean': df[field].mean(),
        #'var': df[field].var(),
        #'sdev': df[field].std(),
        'Pearson`s corr coef': sp.stats.pearsonr(df1['Life expectancy '],df1[field])
    })
#print(df['Life expectancy '])    
for field in fields:
    print(field)
    
#dropping values gives much more promising results in terms of probability p of correlation
#can we predict life expectancy of a country by given data?
    
#Creating regression model for life expectancy
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#print(df1.isnull().any())

result = []
for x in df1.columns:
    if x != 'Life expectancy ':
        result.append(x)
   
X = df1[result].values
y = df1['Life expectancy '].values
#for i in range(50):
    #, random_state=i*10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=50)

# build the model
model = LinearRegression()  
model.fit(X_train, y_train)

print(model.coef_)

y_pred = model.predict(X_test)

#build a new data frame with two columns, the actual values of the test data, 
#and the predictions of the model

df1_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1_head = df1_compare.head(7)
print(df1_head)
print('Mean:', np.mean(y_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("RSME percent of mean:",np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test))

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
chart_regression(y_pred[:50].flatten(),y_test[:50],sort=True)
chart_regression(y_pred[:100].flatten(),y_test[:100],sort=True)
chart_regression(y_pred[:200].flatten(),y_test[:200],sort=True)   

#second data frame
#Strip non-numeric data
#df2 = df.select_dtypes(include=['int', 'float'])
#headers = list(df2.columns.values)
#fields = []

#using median value to fill NA fields 
#for field in headers:
#    med = df2[field].median()
#    df2[field] = df2[field].fillna(med)
#    print(f" #{field} has na? {pd.isnull(df[field]).values.any()}")
#    fields.append({
#        'name' : field,
#        #'mean': df2[field].mean(),
#        #'var': df2[field].var(),
#        #'sdev': df2[field].std(),
#        'Pearson`s corr coef': sp.stats.pearsonr(df2[field], df2['Life expectancy '])
#    }) 
#for field in fields:
#    print(field)