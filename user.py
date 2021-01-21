#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The user file includes the data preprocessing and function definitions

import os
import numpy as np
import pandas as pd
from math import log
from math import sqrt
from numpy import matrix
from numpy import array

## Data preprocessing: generate a random dataset a with n samples and d variables
def gen_data(n, d):
    global a
    global S
    global E
       
    temp = [[(np.random.normal(loc=0.0, size=None)) for i in range(d)] for j in range(n)]
    a=np.matrix(np.array(temp))
    S = [[(a[i].T * a[i]) for i in range(n)]] # change the set
    S = S[0]
    E=np.eye(d, dtype=int)
    E=np.matrix(np.array(E))
 

# primal objective function logdet(X)
def f(x,d):		
    val = 0.0
    indexN = np.flatnonzero(x)
    for i in indexN:
        val = val + x[i]*S[i]
  
    val=np.linalg.slogdet(val) 
    val=val[1]
    return val
    
# Update the inverse matrix by adding a rank-one matrix
def upd_inv_add(X, Xs, opti): 
    Y = 0.0
    Y =  X + S[opti]
    
    Ya = Xs*a[opti].T
    Ys = Xs - 1/(1+(a[opti]*Ya)[0,0])*(Ya*Ya.T)   
    return Y, Ys

# Update the inverse matrix by minusing a rank-one matrix
def upd_inv_minus(X, Xs, opti):
    Y = 0.0 
    Y =  X - S[opti]
    
    Ya = Xs*a[opti].T
    Ys = Xs + 1/(1-(a[opti]*Ya)[0,0])*(Ya*Ya.T)  
    return Y, Ys


# rank-one update for greedy if the number of samples is less than the dimension
def srankone(X, Xs,indexN,n):
    nx = len(indexN)
    y = [0]*nx
    j=0
    opti = 0.0
    Y = 0.0
    Ya = 0.0
    Yb = 0.0
    Yc = 0.0
    Ys = 0.0
    
    for i in indexN:
        y[j] = (a[i]*(E-Xs*X)*a[i].T).tolist()[0][0]
        j = j+1
        
    opti  = indexN[y.index(max(y))]
    
    Y = X + S[opti]
    
    Ya = Xs*a[opti].T    
    Yb = (E-Xs*X)*a[opti].T
    Yc = 1/(a[opti]*Yb)[0,0]
    
    Ys = Xs - Yc*(Ya*Yb.T)-Yc*(Yb*Ya.T)+(Yc**2)*(1+(a[opti]*Ya)[0,0])*(Yb*Yb.T)
    
    return Y,Ys,opti 
   
# rank-one update for greedy if the number of samples is greater than the dimension   
def nrankone(Xs, indexN,n):
    nx = len(indexN)
    y = [0]*nx
    j=0
    opti = 0
    Ya = 0.0
    Ys = 0.0

    for i in indexN:
        y[j] = (a[i]*Xs*a[i].T).tolist()[0][0]
        j = j+1
    
    opti = indexN[y.index(max(y))]
    
    Ya = Xs*a[opti].T
    Ys = Xs - 1/(1+(a[opti]*Ya)[0,0])*(Ya*Ya.T)
    
    return Ys, opti



## The rank one update for local search
def findopt(X, Xs, i, indexN,n,val):
    Y=0.0
    Ys=0.0
    
    Y, Ys = upd_inv_minus(X, Xs, i)
    
    temp = []
    for j in indexN:
        temp.append(1 + a[j]*Ys*a[j].T)
        
    tempi = np.argmax(temp)
    opti = indexN[tempi]
        
    val = val - log(1 + a[i]*Ys*a[i].T) + log(1+a[opti]*Ys*a[opti].T)
       
    return Y, Ys, opti, val


        
 
 
