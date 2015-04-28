#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV

CPUS=multiprocessing.cpu_count()

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
    X_train, y_train = train[:,:-1],train[:,-1]
    X_test, y_test =test[:,:-1], test[:,-1]
    X_train, X_test = normalize(X_train, X_test)
    return X_train , y_train, X_test, y_test

#Cross-validate to choose between linear, poly and RBF.
def kernel_cv(X,y,cv=3):
    kernels=["linear","poly","rbf"]
    scores={}
    for x in kernels:
        clf = SVC(kernel=x) #', C=1)
        score = cross_val_score(clf, X, y, cv=cv)
        scores[x]=sum(score)/cv.n_iter
    return max(scores, key=scores.get)
    
#Create the heatmap for RBF
def heat_map(X,y,cv=2,n_jobs=CPUS):
    C_range = np.logspace(0, 7, 8)
    gamma_range = np.logspace(-3, 3, 7)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=n_jobs)
    grid.fit(X, y)
    return grid,gamma_range,C_range


##############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis

if __name__ == "__main__":
    X , y, X_test, y_test = pamap_easy()
    cv = StratifiedShuffleSplit(y, n_iter=5, train_size=0.5, test_size=0.5)
    print "1. Cross-validate to choose between linear, poly and RBF:",
    cv = StratifiedShuffleSplit(y, n_iter=3, train_size=0.01, test_size=0.005)
    print kernel_cv(X,y,cv=cv)
    print "2. Creating the heatmap, for speedup on very small datasets - results may vary..."
    grid,gamma_range,C_range=heat_map(X,y,cv=cv)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    plt.figure(figsize=(8, 7))
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
    print("3. the best score: {} with parameters, gamma: {}, C:{}".format(grid.best_score_,
           grid.best_params_["gamma"], grid.best_params_["C"]))

   


