#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def normalize(X_train, X_test):
    for i in range(X_train.shape[1]):
        _min=min(X_train[:,i].min(),X_test[:,i].min())
        _max=max(X_train[:,i].max(),X_test[:,i].max())
        X_train[:,i] = (X_train[:,i] - _min) / (_max - _min)
        X_test[:,i] = (X_test[:,i] - _min) / (_max - _min)
    return X_train, X_test

def pamap_easy():
    train=np.loadtxt("pamap_easy.train.txt")
    test=np.loadtxt("pamap_easy.test.txt")
    #dam tam zpocatku nuly, abych si vystacila s jedinou tetou
    X_train, y_train = train[:,:-1],train[:,-1]
    X_test, y_test =test[:,:-1], test[:,-1]
    X_train, X_test = normalize(X_train, X_test)
    return X_train , y_train, X_test, y_test

X , y, X_test, y_test = pamap_easy()

#Cross-validate to choose between linear, poly and RBF.
def kernel_cv(X,y,cv=4):
    kernels=["linear","poly","rbf"]
    scores={}
    for x in kernels:
        clf = SVC(kernel=x) #', C=1)
        score = cross_val_score(clf, X, y, cv=cv)
        scores[x]=sum(scores)/10
    return max(scores, key=scores.get)
    
#Create the heatmap for RBF
def heat_map(X,y,cv=2):
    C_range = np.logspace(0, 7, 8)
    gamma_range = np.logspace(-3, 3, 7)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
ip


##############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis



print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
