#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("subject101.dat.gz")
activityID=data[:,1]
heartRate=data[:,2]
