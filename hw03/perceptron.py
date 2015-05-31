#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

#Conversions

unit_step = lambda x: 0 if x < 0 else 1

def artificial_data():
    names = ("size,colour,shape,object").split(',')
    train=pd.read_csv("artificial_separable_train.csv", names=names,delimiter=",")
    test=pd.read_csv("artificial_separable_test.csv", names=names,skiprows=1,delimiter=",")
    
    X_train=train.drop("object",1).T.to_dict().values()
    X_test=test.drop("object",1).T.to_dict().values()
    v = DictVectorizer(sparse=False)
    X_train_encoded=v.fit_transform(X_train)
    X_test_encoded=v.transform(X_test)

    y_train=train.object.apply( lambda x: 0 if x=="other" else 1)
    y_test=test.object.apply( lambda x: 0 if x=="other" else 1)
   
    return X_train_encoded, X_test_encoded, y_train, y_test

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
    X_train_encoded=v.fit_transform(X_train)
    X_test_encoded=v.transform(X_test)
    
    return X_train_encoded, X_test_encoded, y_train, y_test

def perceptron_learn(X, y, eta=0.2, n=1000):
    w=np.random.rand((X.shape[1]))
    for i in xrange(n):
        n=np.random.randint(0,X.shape[0]-1)
        #x, expected = choice(training_data)
        result = np.dot(w, X[n])
        error = y[n] - unit_step(result)
        w += eta * error * X[n]
    return w

def perceptron_test(X,y,w):
    return len([n for n in range(len(X)) if unit_step(np.dot(X[n], w)) == y[n]]) / float(len(X))
 
def evaluate_dataset(ds,n=1000):
    X_train,X_test,y_train,y_test=ds
    w=perceptron_learn(X_train,y_train,n=n)
    print "Training data accuracy:",perceptron_test(X_train,y_train,w)
    print "Test data accuracy:",perceptron_test(X_test,y_test,w)

if __name__ == "__main__":
     print "Artificial_data:"
     evaluate_dataset(artificial_data(),n=1000)
     print "Adult_data:"
     evaluate_dataset(adult_data(),n=50000)
     
     
      
   
#
#for x, _ in training_data:
#    result = dot(x, w)
#    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


#blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
