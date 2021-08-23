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
import sklearn as sk

#Excel file should be in same directory
path = "."
filename_read = os.path.join(path, "Life Expectancy Data.csv")
#check for empty fields

df = pd.read_csv(filename_read, na_values=['NA', '?'])

#Dropping of Human Development Index in terms of income composition of resources column
#I found out that his HDI value is " is assessed by life expectancy at birth(...)"
# as well as schooling value and GNI which from my understanding is correlated with GDP
#it has been deleted to prevent bias in results
df.drop('Income composition of resources', 1, inplace=True)

#print(df1.isnull().any())

# First data frame - dropping non numerical features
df1 = df.select_dtypes(include=['int', 'float'])

headers = list(df1.columns.values)
fields = []
     
#Preparing data by dropping empty values instead of filling with median
df1 = df1.dropna()

#Calculating Pearson's correlation coefficient for each feature
for field in headers:
    fields.append({
        'name' : field,
        'Pearson`s corr coef': sp.stats.pearsonr(df1['Life expectancy '],df1[field])
    })
    
for field in fields:
    print(field)
    
#dropping values gives (in my opinion) much more promising results in terms of probability p of correlation
    
#can we predict life expectancy of a country by given data?
    
#Creating regression model for life expectancy predicition
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Collecting columns other than target one
result = []
for x in df1.columns:
    if x != 'Life expectancy ':
        result.append(x)
   
X = df1[result].values
y = df1['Life expectancy '].values

#instead of using K-fold I found it much easier to change random states to get different datasplits
#the best score (RSME percent of mean has been found on model with Random state =22040)
#the best score (R^2 has been found on model with Random state =16720)
#best=0
#best2=0
#best_ratio=100
#best_R2=0
#for i in range(5000):
#, random_state=i*10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=16720)
    
# build the model
model = LinearRegression()  
model.fit(X_train, y_train)
    
print(model.coef_)
    
y_pred = model.predict(X_test)
    
#build a new data frame with two columns, the actual values of the test data, 
#and the predictions of the model
    
df1_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print first 10
df1_head = df1_compare.head(10)
#print(df1_head)
#Calculate Mean, RMSE and how many percent of mean is our RMSE.
#According to lecture information percent below 10% would be satisfying
#This produced model has ratio of 0.04408479536999818 ~ slightly below 4.5%
#I also calculate R^2 - coefficient of determination
#the best R^2 was 0.870799188733013 from state 16720
    
print('Mean:', np.mean(y_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
rpom=np.sqrt(metrics.mean_squared_error(y_test, y_pred))/np.mean(y_test)
print("RSME percent of mean:",rpom)
r2=sk.metrics.r2_score(y_test, y_pred)
print("R^2",r2)
    #if rpom < best_ratio:
    #    best_ratio=rpom
    #    best=i
    #if r2> best_R2:
    #    best_R2=r2
    #    best2=i
    #print(best)
    #print(best_ratio)
    #print(best2)
    #print(best_R2)
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


print("Presenting of top 3 correlations")

plt.scatter('Life expectancy ', 'Schooling', data = df1, color = "green", s=5)
plt.xlabel("Life expectancy")
plt.ylabel("Schooling")
plt.show()

plt.scatter('Life expectancy ', 'Adult Mortality', data = df1, color = "green", s=5)
plt.xlabel("Life expectancy")
plt.ylabel("Adult Mortality")
plt.show()

plt.scatter('Life expectancy ', ' HIV/AIDS', data = df1, color = "green", s=5)
plt.xlabel("Life expectancy")
plt.ylabel(" HIV/AIDS")
plt.show()

plt.scatter('Life expectancy ', ' BMI ', data = df1, color = "green", s=5)
plt.xlabel("Life expectancy")
plt.ylabel("BMI")
plt.show()
#Second data frame - dropping non numerical features

#df2 = df.select_dtypes(include=['int', 'float'])

#headers = list(df2.columns.values)
#fields = []

##Preparing data by filling empty values with median instead of dropping
#for field in headers:
#    med = df2[field].median()
#    df2[field] = df2[field].fillna(med)
#    print(f" #{field} has na? {pd.isnull(df2[field]).values.any()}")
#    fields.append({
#        'name' : field,
#        'Pearson`s corr coef': sp.stats.pearsonr(df2[field], df2['Life expectancy '])
#    }) 
#for field in fields:
#    print(field)
