#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pdb

def adult_data():
    names = ("age, workclass, fnlwgt, education, education-num, "
             "marital-status, occupation, relationship, race, sex, "
             "capital-gain, capital-loss, hours-per-week, "
             "native-country, income").split(', ')
    train=pd.read_csv("adult.data", names=names,delimiter=", ").dropna()
    test=pd.read_csv("adult.test", names=names,skiprows=1,delimiter=", ").dropna()
    #
    y_train=train.income.apply( lambda x: 0 if x=="<=50K" else 1)
    y_test=test.income.apply( lambda x: 0 if x=="<=50K." else 1)
    X_train=train.drop("income",1).T.to_dict().values()
    X_test=test.drop("income",1).T.to_dict().values()
    #
    v = DictVectorizer(sparse=False)
    X_train_encoded=v.fit_transform(X_train)
    X_test_encoded=v.transform(X_test)
    #
    return X_train_encoded, X_test_encoded, y_train, y_test

def credit_data():
    train_data=np.loadtxt("credit.train.txt",delimiter=",")
    test_data=np.loadtxt("credit.test.txt",delimiter=",")
    X_train,y_train=train_data[:,:-1],train_data[:,-1]
    X_test,y_test=test_data[:,:-1],test_data[:,-1]
    return X_train, X_test,y_train, y_test

def cloud_data():
    train_data=np.loadtxt("clouds_dataset/clouds1k_train.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    test_data=np.loadtxt("clouds_dataset/clouds1k_test.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    X_train, y_train = train_data[:,1:-1],train_data[:,-1]
    X_test, y_test = test_data[:,1:-1],test_data[:,-1]
    return X_train, X_test, y_train, y_test

#aesop data
def aesop_data():
    header=open("aesop.data","r").readline().strip().split(",")
    data=pd.read_csv("aesop.data",skiprows=1,names=header)
    #
    y = data.student
    X = data.drop("student",1).T.to_dict().values()
    v = DictVectorizer(sparse=False)
    X_encoded=v.fit_transform(X)
    return train_test_split(X_encoded, y, test_size=.4)


NAMES = ["Nearest Neighbors", 
         "Random Forest", 
         "AdaBoost",
         "RandomForestClassifier"]

CLASSIFIERS = [KNeighborsClassifier(3),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               AdaBoostClassifier(),
               GradientBoostingClassifier(init=None, learning_rate=1.0, loss='deviance',
               max_depth=1, max_features=None, min_samples_leaf=1,
               min_samples_split=2, n_estimators=100, random_state=0,
               subsample=1.0, verbose=0)]

def evaluate_dataset(ds,label):
    print label
    X_train, X_test, y_train, y_test = ds
    for name,clf in zip(NAMES,CLASSIFIERS):
       clf.fit(X_train, y_train)
       score = clf.score(X_test, y_test)
       print "\t",name,score
    
if __name__ == '__main__':
    print "Accuracy on a particular data set:"
    evaluate_dataset(adult_data(),"- adult:")
    evaluate_dataset(credit_data(),"- credit data:")
    evaluate_dataset(cloud_data(),"- cloud:")
    evaluate_dataset(aesop_data(),"- aesop:")
		        
