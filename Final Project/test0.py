# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:31:36 2018

@author: Harsh Kava
"""



import pandas as pd
import numpy as np


df=pd.read_csv('Downloads\world_cup_results.csv')

df.groupby('Round')['HomeGoals'].mean()
df.groupby('Round')['HomeGoals'].transform(lambda x:x.mean())
df.groupby('Round').transform(lambda x:x['HomeGoals'].mean())
df.groupby('Round')['HomeGoals'].transform(np.mean())

df1=pd.read_csv('Downloads\world_cup_results.csv',usecols = [1,2])
customes_list=[1,2,3,4,5]

for a, b in enumerate(customes_list):
    print(a)
    
    
    def foo():
        print(2)
        
print(type(foo()))

library(ts)
myts <- ts(myvector, start=c(2009, 1), end=c(2014, 12), frequency=12) 




df = pd.DataFrame([None,None, 3.0,4.0,None,10.0],[None,None, 3.0,4.0,None,10.0])



df.fillna(500)