#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the greedy algorithm

import user 
import numpy as np
import datetime


# assign local names to functions in the user file
gen_data = user.gen_data
srankone = user.srankone
f = user.f
nrankone = user.nrankone

# Function grd needs input n, d and s; outputs the objectve value, 
# solution, matrix \sum_{i in S}v_i*v_i^T and its inverse, 
# and running time of the greedy algorithm
def grd(n, k, d):

    start = datetime.datetime.now()
    
    c = 1
    x = [0]*n # chosen set
    y = [1]*n # unchosen set
    indexN = np.flatnonzero(y)
     
    gen_data(n, d) # load data
    
    
    gen_data(n,d)    
    index = 0
    X=np.zeros([d,d])
    Xs=np.zeros([d,d])
    Y=np.zeros([d,d])
    Ys=np.zeros([d,d])
 
 
    while c < k+1:
        if c<=d:
            Y,Ys,index = srankone(X,Xs,indexN,n)
            X =Y
            Xs=Ys
        else:
            Ys,index = nrankone(Xs,indexN,n)
            Xs = Ys
                        
        x[index] = 1
        y[index] = 0
        indexN = np.flatnonzero(y)
        
        c = c + 1
        [a,b] = np.linalg.eigh(Xs) 
        
    grdx = x # greedy solution
    grdf = f(x,d) # greedu value
    
    end = datetime.datetime.now()
    gtime = (end-start).seconds
    
    return  grdf,  grdx, Xs.I, Xs, gtime


