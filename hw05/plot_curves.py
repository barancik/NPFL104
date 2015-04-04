#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier #, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import defaultdict

def credit_data():
    train_data=np.loadtxt("credit.train.txt",delimiter=",")
    test_data=np.loadtxt("credit.test.txt",delimiter=",")
    X_train,y_train=train_data[:,:-1],train_data[:,-1]
    X_test,y_test=test_data[:,:-1],test_data[:,-1]
    return X_train, X_test, y_train, y_test

def cloud_data():
    train_data=np.loadtxt("clouds_dataset/clouds1k_train.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    test_data=np.loadtxt("clouds_dataset/clouds1k_test.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    X_train, y_train = train_data[:,1:-1],train_data[:,-1]
    X_test, y_test = test_data[:,1:-1],test_data[:,-1]
    return X_train, X_test, y_train, y_test

def aesop_data():
    header=open("aesop.data","r").readline().strip().split(",")
    data=pd.read_csv("aesop.data",skiprows=1,names=header)
    y = data.student
    X = data.drop("student",1).T.to_dict().values()
    v = DictVectorizer(sparse=False)
    X_encoded=v.fit_transform(X)
    return train_test_split(X_encoded, y, test_size=.4)
    
def plot(title,data):
    arr, train_error, test_error = data
    plt.figure()
    plt.xkcd()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.plot(arr,train_error,'o-', color="r",label="Training error")
    plt.plot(arr,test_error, 'o-', color="g",label="Test error")
    plt.xscale("log")
    plt.legend(loc="best")

def avg_error(ds, ticks=30, repetitions=10):
    X_train, X_test, y_train, y_test = ds
    estimator=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    train_error, test_error = defaultdict(list), defaultdict(list)
    n=X_train.shape[0]
    arr=np.logspace(np.log2(10),np.log2(n),base=2,num=ticks)
    for a in range(repetitions):
        X_train, y_train = shuffle(X_train, y_train)
        for x in arr:
	    clf=estimator.fit(X_train[:x,:],y_train[:x])
	    train_error[a].append(1 - estimator.score(X_train[:x,:],y_train[:x]))
	    test_error[a].append(1 - estimator.score(X_test,y_test))
    avg_train_error=[sum([train_error[x][y] for x in range(repetitions)])/repetitions for y in range(ticks)]
    avg_test_error=[sum([test_error[x][y] for x in range(repetitions)])/repetitions for y in range(ticks)]
    return arr,avg_train_error, avg_test_error   
   
if __name__ == '__main__':
    plot("Credit data",avg_error(credit_data(),repetitions=50))
    plot("Cloud data",avg_error(cloud_data(),repetitions=50))
    plot("Aesop data",avg_error(aesop_data(),repetitions=50))
    plt.show()
