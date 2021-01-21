#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The implementation of the local search algorithm

import greedy
import datetime
import user
import numpy as np
import pandas as pd
import math

# assign local names to functions in the user and greedy file
findopt =  user.findopt
upd_inv_add = user.upd_inv_add
upd_inv_minus = user.upd_inv_minus
gen_data = user.gen_data
grd = greedy.grd
f = user.f

# Function localsearch needs input n, d and s; outputs the objectve values, 
# and running time of the greedy and local search algorithms
def localsearch(n, k, d):    
    start = datetime.datetime.now()
    
    ## greedy algorithm
    grdval, bestx, X, Xs, gtime = grd(n, k, d)
    print("The running time of Greedy algorithm = ", gtime)
    print("The output value of Greedy algorithm = ", math.exp(grdval/d))
    
    sel = [i for i in range(n) if bestx[i] == 1] # chosen set
    t = [i for i in range(n) if bestx[i] == 0] # unchosen set                 
  
    ## local search    
    Y = 0.0
    Ys = 0.0 
    fval = 0.0
    optimal = False
    bestf = grdval

    while(optimal == False):
        optimal = True
        
        for i in sel :
            Y = 0.0
            Ys = 0.0
            tempX = X
            tempXs = Xs
            
            Y, Ys, index,fval = findopt(tempX, tempXs, i, t, n, bestf)
            
            if fval > bestf:
                print("The current objective value of Local Sarch algorithm = ", math.exp(fval/d))
                optimal = False                
                bestx[i] = 0
                bestx[index] = 1 # update solution                 
                bestf = fval # update the objective value
                
                X, Xs = upd_inv_add(Y, Ys, index) # update the inverse
                
                sel = np.flatnonzero(bestx) # update chosen set
                t = [i for i in range(n) if bestx[i] == 0] # update the unchosen set
                break

    end = datetime.datetime.now()
    time = (end - start).seconds         
    print('The outpue value of Local Sarch algorithm = ', math.exp(bestf/d))
       
    return grdval, gtime, bestf,  time
