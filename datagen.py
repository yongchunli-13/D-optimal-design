#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Implementation of greedy and local search Algorithms

import pandas as pd
import numpy as np
import local_search
import math

# assign local names
localsearch  = local_search.localsearch


# parameters
n = 10000 #number of experiments
d = 200 # dimension

loc=0
df = pd.DataFrame(columns=('n','k','grdval', 'gtime','lsval','ltime')) 
# grdval: objective value of greedy; gtime: runnig time of greedy
# lsval: objective value of local search; ltime: runnig time of local search

# k is the number of selected experiments
for k in range(d+1, d+100, 10):
    print("this is case ", loc+1)
    grdval, gtime, lsval, ltime  = localsearch(n, k, d) 
    df.loc[loc] = np.array([n, k, math.exp(grdval/d), gtime, math.exp(lsval/d), ltime])
    loc = loc+1  

