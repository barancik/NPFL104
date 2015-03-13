#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
from scikits.statsmodels.tools import categorical


a = np.array( ['a', 'b', 'c', 'a', 'b', 'c'])
b = categorical(a, drop=True)

LABELS={1:"lying",4:"walking",5:"running"}
data=np.loadtxt("subject101.dat.gz")

def plot(array,label):
    # Plots histosgram of the array and fitted gaussian
    (mu, sigma) = norm.fit(array)
    n, bins, patches = plt.hist(array,20,alpha=0.8,normed=True,label=label)
    y = mlab.normpdf(bins,mu,sigma)
    l = plt.plot(bins,y,'r--',linewidth=2)

activities={x:[] for x in LABELS}

# Extracting data
for x in data:
    if x[1] in activities.keys() and not np.isnan(x[2]):
        activities[x[1]].append(x[2])

# Main loop
for x in activities:
     plot(activities[x],LABELS[x])

plt.ylabel('Count')
plt.xlabel('Heart rate')
plt.legend() 
plt.show()
      
                   
