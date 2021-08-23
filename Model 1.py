# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:29:20 2021

@author: wkxxx
"""

import os
import pandas as pd
import numpy as np
import scipy as sp

path = "."

filename_read = os.path.join(path, "Life Expectancy Data.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

headers = list(df.columns.values)
fields = []

#using median value    
for field in headers:
    med = df[field].median()
    df[field] = df[field].fillna(med)
    print(f" has na? {pd.isnull(df[field]).values.any()}")
    fields.append({
        'name' : field,
        #'mean': df[field].mean(),
        #'var': df[field].var(),
        #'sdev': df[field].std(),
        'Pearson`s corr coef': sp.stats.pearsonr(df[field], df['Life expectancy '])
    }) 
#print(df['Life expectancy '])    
for field in fields:
    print(field)
    
filename_read = os.path.join(path, "Life Expectancy Data.csv")
df = pd.read_csv(filename_read, na_values=['NA', '?'])

# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

headers = list(df.columns.values)
fields = []
     
#repeating with dropping values
for field in headers:
    df = df.dropna()
    fields.append({
        'name' : field,
        #'mean': df[field].mean(),
        #'var': df[field].var(),
        #'sdev': df[field].std(),
        'Pearson`s corr coef': sp.stats.pearsonr(df[field], df['Life expectancy '])
    })
#print(df['Life expectancy '])    
for field in fields:
    print(field)
    #dropping values gives much more promising results in terms of probability p of correlation
    #