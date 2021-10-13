#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:19:32 2021

@author: abdul
"""


import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



data1= pd.read_excel('Data for Case Study.xlsx', sheet_name="Sheet1")

data1.drop(['Control Count','Date Replied','Unique identifier'],axis=1, inplace=True)

headers=list(data1)

data1 = data1.rename(columns={'Likely to recommend Online Site "MY SITE"': 'Y'})


df1=data1.iloc[:,0:28]
df1=df1.fillna(value=0)
df2=data1.iloc[:,28:len(data1.columns)]
        
data2=pd.concat([df1,df2],axis=1)



for col in range(2,28):
    data2.iloc[:,col] = data2.iloc[:,col].astype('category')
    
    
        
def cat_num(df, col):
    s1 = pd.Series(df[col])
    labels1, levels1 = pd.factorize(s1)
    df[col] = labels1
    return df

def drop_rows_with_na(df,na_limit):
    df= df.dropna(thresh=(len(df.columns)-int(na_limit*len(df.columns))))    
    #na_list=df.isnull().sum(axis=1)
    return df

def drop_cols_with_na(df,na_limit):
    df=df.loc[:, df.isnull().mean() < na_limit]
    return df

def split_data(x,y):
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    return(X_train,X_test,y_train,y_test)


def cat_missing(df,col):
    df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def filling_mean_values(df):
        df=df.fillna(df.mean())
        return df
    
def normalize(df):
        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df

def class_w(y_train):
        counts= y_train.value_counts()
        rev=1/counts
        vv=dict(rev)
        class_weights=vv
        return class_weights

def one_hot_encoding(y):
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()
    return y

   
from keras.layers.core import Lambda


def neural_network(input_dim, output_dim, n_x_train_filled, n_x_test_filled, y_train, y_test):
    model = Sequential()
    model.add(Dense(300, input_dim=input_dim, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_dim, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(n_x_train_filled, y_train, validation_data=(n_x_test_filled, y_test), 
                        epochs=100, batch_size=128, verbose=1)
    
    y_pred = model.predict(n_x_test_filled)
    
    return y_pred
    




def prediction(y_pred, y_test):
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
        
        
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
    
    cm=classification_report(test, pred)

    return (print(cm))


data2=drop_rows_with_na(data2,0.4)
data3=drop_cols_with_na(data2,0.4)

y=data3['Y']

data3.drop(['Y'],axis=1, inplace=True)

x=data3



#plots
corr_matrix = x.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


dropped_cols = [column for column in upper.columns if any(upper[column] > 0.85)]

# Drop features 
x.drop(dropped_cols, axis=1, inplace=True)

x1=x



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for i in [i for i in x1.columns if x1[i].dtype in numerics]:
    x1[i] = np.log(x1[i])


x1.replace([np.inf, -np.inf], 5000, inplace=True)


x1['Y']=y

new_df=pd.DataFrame([])
for j in range(0,len(x1)):
    row=(x1.iloc[j,:])
    print(j)
    if 5000 not in row.values:
        new_df=new_df.append(row,ignore_index=True)
    



def outliers(z):
    #sns.boxplot(data=z)
    z=np.log(z)
    Q1 = z.quantile(0.25)
    Q3 = z.quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 

    filter = ((z >= Q1 - 1.5 * IQR) & (z<= Q3 + 1.5 *IQR)) | (z.isna())
    z1=z.loc[filter]  
    return(len(z)-len(z1))
    
    


X=new_df[new_df.columns[:-1]]
y=new_df['Y']    
y=y/10



X_train,X_test,y_train,y_test=split_data(X,y)
    
x_train_filled=filling_mean_values(X_train)
x_test_filled=filling_mean_values(X_test)

x_train_filled=cat_missing(x_train_filled,'Proportion of purchase made with "MY SITE"')
x_train_filled=cat_missing(x_train_filled,'Future proportion of purchase made at "MY SITE"')

x_test_filled=cat_missing(x_test_filled,'Proportion of purchase made with "MY SITE"')
x_test_filled=cat_missing(x_test_filled,'Future proportion of purchase made at "MY SITE"')



x_train_filled=cat_num(x_train_filled,'Status')
x_train_filled=cat_num(x_train_filled,'Proportion of purchase made with "MY SITE"')
x_train_filled=cat_num(x_train_filled,'Future proportion of purchase made at "MY SITE"')


x_test_filled=cat_num(x_test_filled,'Status')
x_test_filled=cat_num(x_test_filled,'Proportion of purchase made with "MY SITE"')
x_test_filled=cat_num(x_test_filled,'Future proportion of purchase made at "MY SITE"')

    
n_x_train_filled=normalize(x_train_filled)
n_x_test_filled=normalize(x_test_filled)

#class_weights=class_w(y_train)
    
#y_train = one_hot_encoding(pd.DataFrame(y_train))
#y_test = one_hot_encoding(pd.DataFrame(y_test))

input_dim=len(n_x_train_filled.columns)
output_dim=1
#output_dim=1
 

y_pred=neural_network(input_dim, output_dim, n_x_train_filled, n_x_test_filled, y_train, y_test)


cm_nn=prediction(y_pred,y_test)












