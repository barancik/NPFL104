#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import defaultdict
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def pamap_easy():
    train=np.loadtxt("pamap_easy.train.txt")
    test=np.loadtxt("pamap_easy.test.txt")
    X_train, y_train = train[:,:-1],train[:,-1]
    X_test, y_test =test[:,:-1], test[:,-1]
    return np.vstack((X_train,X_test)), np.hstack((y_train, y_test))

def _new_clusters(X,centroids):
    clusters=defaultdict(list)
    for point in X:
        closest=min([[i,np.linalg.norm(point-x)] for i,x in enumerate(centroids)],key=lambda z:z[1])[0]
        clusters[closest].append(point)
    centroids=[np.mean(x,axis=0) for x in clusters.values()]
    inertia=sum([sum([np.linalg.norm(centroids[y]-x) for x in clusters[y]]) for y in clusters.keys()])
    return centroids,inertia

def _converged(new,old):
    return (set([tuple(a) for a in new]) == set([tuple(a) for a in old]))

def k_means(X,n_centroids):
    centroids=random.sample(X,n_centroids)
    old_centroids=random.sample(X,n_centroids)
    while not _converged(centroids,old_centroids):
        old_centroids=centroids
        centroids, inertia =_new_clusters(X,centroids)
        print "Inertia:",inertia
    return np.array([min([[i,np.linalg.norm(point-x)] for i,x in enumerate(centroids)],
			 key=lambda z:z[1])[0] for point in X]), inertia
  
def print_scores(labels,guessed_labels, inertia, name, data):
    print("% 9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"
          % (name, inertia,
             metrics.homogeneity_score(labels, guessed_labels),
             metrics.completeness_score(labels, guessed_labels),
             metrics.v_measure_score(labels, guessed_labels),
             metrics.adjusted_rand_score(labels, guessed_labels),
             metrics.adjusted_mutual_info_score(labels,  guessed_labels),
             metrics.silhouette_score(data,guessed_labels,metric='euclidean', sample_size=300)))

def bench_k_means(estimator, name, data, labels):
    estimator.fit(data)
    print_scores(labels,estimator.labels_, estimator.inertia_, name, data)


if __name__ == "__main__":
    X , y = pamap_easy()
    n_centroids=len(set(y.T))
    labels,inertia=k_means(X,n_centroids)
    
    print(79 * '_')
    print("init\t\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    print_scores(y, labels, inertia, "k_means", X)

    bench_k_means(KMeans(init='k-means++', n_clusters=n_centroids, n_init=10),
              name="k-means++", data=X, labels=y)

    bench_k_means(KMeans(init='random', n_clusters=n_centroids, n_init=10),
              name="random", data=X, labels=y)

    pca = PCA(n_components=n_centroids).fit(X)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_centroids, n_init=1),
              name="PCA-based", data=X, labels=y)
    
    print(79 * '_')

