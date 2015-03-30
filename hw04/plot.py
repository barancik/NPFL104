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

def adult_data():
    names = ("age, workclass, fnlwgt, education, education-num, "
             "marital-status, occupation, relationship, race, sex, "
             "capital-gain, capital-loss, hours-per-week, "
             "native-country, income").split(', ')
    train=pd.read_csv("adult.data", names=names,delimiter=", ").dropna()
    test=pd.read_csv("adult.test", names=names,skiprows=1,delimiter=", ").dropna()
    y_train=train.income.apply( lambda x: 0 if x=="<=50K" else 1)
    y_test=test.income.apply( lambda x: 0 if x=="<=50K." else 1)

    X_train=train.drop("income",1).T.to_dict().values()
    X_test=test.drop("income",1).T.to_dict().values()

    v = DictVectorizer(sparse=False)
    X_train_encoded=v.fit_transform(train_X)
    X_test_encoded=v.transform(test_X)
    return train_encoded_X, train_y, test_encoded_X, test_y

def credit_data():
    train_data=np.loadtxt("credit.train.txt",delimiter=",")
    test_data=np.loadtxt("credit.test.txt",delimiter=",")
    X_train,y_train=train_data[:,:-1],train_data[:,-1]
    X_test,y_test=test_data[:,:-1],test_data[:,-1]
    return X_train, y_train, X_test, y_test

def cloud_data():
    train_data=np.loadtxt("clouds_dataset/clouds1k_train.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    test_data=np.loadtxt("clouds_dataset/clouds1k_test.csv",skiprows=1,converters = {0: lambda x:0},delimiter=", ")
    X_train, y_train = train_data[:,1:-1],train_data[:,-1]
    X_test, y_test = test_data[:,1:-1],test_data[:,-1]
    return X_train, y_train, X_test, y_test

#aesop data
def aesop_data():
    data=pd.read_csv("aesop",skiprows=1)


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
#classifiers = [adult_data(), 
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

datasets=[ adult_data(),credit_data(),cloud_data()]

for name,clf in zip(names,classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print name,score

    
