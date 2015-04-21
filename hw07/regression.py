#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import *
import pdb
from sklearn.feature_extraction import DictVectorizer

#from matplotlib.colors import Normalize
#from sklearn.svm import SVC
#from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris

def jcost(theta,x,y):
    n=len(y)
    return sum([(hypothesis(theta,x[i]) - y[i])**2 for i in range(n)])/(2*n) #(2*len(y))

def hypothesis(theta,x):
    return sum(theta*x)

def gradient_descent(x,y,alpha=0.0005,end_convergence=0.00001):
    converged=False
    idx=0
    n=x.shape[0]
    x=np.hstack((np.ones((n,1)),x))
    m=x.shape[1]
    theta=np.random.random(m)
    J = jcost(theta,x,y)
    while not converged:
        delta=[alpha*sum([(hypothesis(theta,x[i])-y[i])*x[i,j] for i in range(n)])/n for j in range(m)]
        theta-=delta
        # mean squared error
        new_J=jcost(theta,x,y)
        if abs(J-new_J) <= end_convergence:
            converged = True
        J = new_J   # update error 
        idx += 1  # update iter
        if idx % 100 == 0:
            print "{}. iteration - mean square error: {}".format(idx,J)
    return theta

def evaluate(X,y,theta):
    return jcost(theta,np.hstack((np.ones((X.shape[0],1)),X)),y)


def artificial_data():
    train=np.loadtxt("artificial_2x_train.tsv")
    test=np.loadtxt("artificial_2x_test.tsv")
    #dam tam zpocatku nuly, abych si vystacila s jedinou tetou
    X_train, y_train = train[:,:-1],train[:,-1]
    X_test, y_test =test[:,:-1], test[:,-1]
    return X_train , y_train, X_test, y_test

def prague_flat_data():
    names = ("velikost, budova, vlastnictvi, stav, pokoje, "
             "zarizeno, sklep, balkon, cena").split(', ')
    train=pd.read_csv("prague_train", names=names,sep="\t", header=None,index_col=False)
    test=pd.read_csv("prague_test", names=names,sep="\t", header=None,index_col=False)
   
    #normalizace
    max_velikost=max(test["velikost"].max(),train["velikost"].max())
    train["velikost"]=train["velikost"].div(max_velikost,axis=0)
    test["velikost"]=test["velikost"].div(max_velikost,axis=0)

    max_cena=max(test["cena"].max(),train["cena"].max())

    X_train=train.drop("cena",1).T.to_dict().values()   
    X_test=test.drop("cena",1).T.to_dict().values()
    y_train=train["cena"].div(max_cena)
    y_test=test["cena"].div(max_cena)
    v = DictVectorizer(sparse=False)
    
    X_train_encoded=v.fit_transform(X_train)
    X_test_encoded=v.transform(X_test)
 
    return X_train_encoded, y_train, X_test_encoded, y_test

def plot_artificial(theta):
    X_train , y_train, X_test, y_test=artificial_data()
    fit_fn = poly1d(theta[::-1])
    plt.plot(X_train,y_train,'yo',X_train,fit_fn(X_train),"--k")
    plt.scatter(X_test,y_test)
    plt.show()

if __name__ == "__main__":
    print "Artificial data"
    X_train , y_train, X_test, y_test=artificial_data()
    theta=gradient_descent(X_train,y_train,alpha=0.0003,end_convergence=0.00001) 
    print "Training data - mean square error:",evaluate(X_train,y_train,theta)
    print "Testing data - mean square error:",evaluate(X_test,y_test,theta)
    print
    print "Prague flat data"
    X_train , y_train, X_test, y_test=prague_flat_data()
    theta=gradient_descent(X_train,y_train,alpha=0.01,end_convergence=0.00001)
    print "Training data - (normalized)  mean square error:",evaluate(X_train,y_train,theta)
    print "Testing data - (normalized) mean square error:",evaluate(X_test,y_test,theta)

