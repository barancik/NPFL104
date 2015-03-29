#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

#adult_data
1 - workclass
3 - education: 
5 - marital-status
6 - occupation
7 - relationship
8 - race
9 -sex
13 - native-country
14 - co chcem, result
categorical_columns=[1,3,5,6,7,8,9,13,14]

data=np.array(np.genfromtxt('dataset.txt', dtype=(int,'S32',int,'S32',int,'S32','S32',\ 'S32','S32','S32',int,int,int,'S32','S32'),delimiter=',',autostrip=True))


#credit_data
train_data=np.loadtxt("credit.train.txt",delimiter=",")
test_data=np.loadtxt("credit.test.txt",delimiter=",")
X_train,y_train=train_data[:,:-1],train_data[:,-1]
X_test,y_test=test_data[:,:-1],test_data[:,-1]

#cloud_data
train_data=np.loadtxt("clouds_dataset/clouds1k_train.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
test_data=np.loadtxt("clouds_dataset/clouds1k_test.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
X_train, y_train = train_data[:,1:-1],train_data[:,-1]
X_test, y_test = test_data[:,1:-1],test_data[:,-1]


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

for name,clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print name,score

    
