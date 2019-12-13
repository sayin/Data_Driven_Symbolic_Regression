#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:45:04 2019

@author: sayin
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg 
#                                         projection='3d' in the call to fig.add_subplot
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################
def AIC(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
#    abss = np.abs(testy - testr.dot(we.real))
#    sse = np.sum(abss)
    m = testy.shape[0]
    aic = m*np.log(sse/m) + 2*k
    
    return aic
   
def BIC(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
#    abss = np.abs(testy - testr.dot(we.real))
#    sse = np.sum(abss)
    m = testy.shape[0]
    bic = m*np.log(sse/m) + k*np.log(m)
    
    return bic    

def BICc(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
#    abss = np.abs(testy - testr.dot(we.real))
#    sse = np.sum(abss)
    m = testy.shape[0]
    bicc = m*np.log(sse/m) + k*np.log((m+2)/24)
    
    return bicc 


def AICc(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
#    abss = np.abs(testy - testr.dot(we.real))
#    sse = np.sum(abss)
    m = testy.shape[0]
    
    cor = 2*(k+1)*(k+2)/(m - k - 2) 
    
    aic = m*np.log(sse/m) + 2*k    
    aicc = aic + cor
    
#    daic  = aicc - aicc.min()
    
    return aicc    


def TrainSTRidge(R, Ut, lam, d_tol, maxit = 50, STR_iters = 50,
                 l0_penalty = None, normalize = 2, 
                 split = 0.8, print_best_tol = True):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a
    training set,then evaluates them  using a loss function on a holdout set.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       
    print(TestY.shape)
    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
#    print d_tol
    tol = d_tol
    
    if l0_penalty == None:
        l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=-1)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):
#        print iter
        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)       

        
        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
            
            test_mse =  mean_squared_error(TestY,  TestR.dot(w.real))  
            cmplx = np.count_nonzero(w)
#            
            aicc  = AICc(TestY, TestR, w, cmplx)
            aic  = AIC(TestY, TestR, w, cmplx)
            bic  = BICc(TestY, TestR, w, cmplx)

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal Parameters:", cmplx, test_mse, tol_best)
     
    
    return  w_best, tol_best, test_mse, cmplx, aicc, aic, bic

def STRidge(X0, y, lam, maxit, tol, normalize, print_results = True):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    

    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d), X.T.dot(y),rcond=-1)[0]
    else: w = np.linalg.lstsq(X,y)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]   
#    print biginds.dtype
    
    # Threshold and continue
    for j in range(maxit):
#        print j
        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
#        print smallinds
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
#                if print_results: print "Tolerance too high -
#                                    all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
#        print biginds
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) +\
                      lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=-1)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]
    
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return  w      
    
#%%
def checksol(data): 
    uux2        = data.uux2.values
    uxu2x       = data.uxu2x.values
    uu3x        = data.uu3x.values
    u2u2x       = data.u2u2x.values
    u4x         = data.u4x.values
    uu2x        = data.uu2x.values
    uer         = data.uer.values  # this is our target, now mapped to Y
    
    predPE =  2.5e-05*uux2 -5.0e-07*uxu2x -2.5e-07*uu3x +1.25e-05*u2u2x +1.25e-09*u4x  -1.25e-05*uu2x    
    
    print("Mean squared error:", mean_squared_error(data.uer, predPE))
    print("R2 score :", r2_score(data.uer,predPE))
    
    plt.figure()
    x = np.arange(0,len(uer))
    plt.plot(x,predPE)       # predictions are in blue
    plt.plot(x,data.uer)       # actual values are in orange
    plt.show()    
    
#doublecheck the data is there
print(os.listdir("./../data_gen/data/."))

# read in the data to pandas
data = pd.read_csv("./../data_gen/data/mod_burgS_data.csv",  encoding='utf-8') #%%Check the solution 
data = data.iloc[::10]
#checksol(data)

#%%
data1 = np.array(data)
theta_ = data1[:,:-1]
uer = data1[:,-1:]  

c = np.ones((theta_.shape[0],1))
theta = np.column_stack((c,theta_))  ## Stack const =1--columnn  to theta

lam1 = data.columns.get_loc("uux2")  + 1    # 2.5e-05uux2
lam2 = data.columns.get_loc("uxu2x") + 1    #-5.0e-07uxu2x
lam3 = data.columns.get_loc("uu3x")  + 1    #-2.5e-07uu3x
lam4 = data.columns.get_loc("u2u2x") + 1    # 1.25e-05u2u2x
lam5 = data.columns.get_loc("u4x")   + 1    # 1.25e-09u4x
lam6 = data.columns.get_loc("uu2x")  + 1    #-1.25e-05uu2x

#%%
key= '''
#Using STRidge to predict the PDE uer =  2.5e-05*uux2 -5.0e-07*uxu2x -2.5e-07*uu3x+ +1.25e-05*u2u2x+ 1.25e-09*u4x  -1.25e-05*uu2x
'''

n_alphas = 20
alphas = np.logspace(-10, -2, n_alphas)
dtol   = 0.00001

otol = np.empty(n_alphas)
test_mse = np.empty(n_alphas)
aicc   = np.empty(n_alphas)
aic    = np.empty(n_alphas)
bic    = np.empty(n_alphas)
aiccn  = np.empty(n_alphas)
cmplx  =  np.empty(n_alphas)

coefs = []
for i, a in enumerate(alphas):
    w, otol[i], test_mse[i], cmplx[i], aicc[i],  aic[i],  bic[i] = TrainSTRidge(theta,uer,a,dtol)
    coefs.append(w) 

coef = np.array(coefs)
cf = coef.reshape(coef.shape[0],coef.shape[1])

param = np.stack((aicc,aic, bic, test_mse, cmplx), axis=1)
temp1 = np.column_stack((param, cf))

opt_param = temp1[temp1[:,2].argsort()]

#%%Psot processing

ax = plt.gca()
for i in range(coef.shape[1]):
    if i == lam1:
        ax.plot(alphas, coef[:,i],'b--',lw=2, label = '$uu_x^2$')    #2.5e-05uux2
    elif i== lam2:
        ax.plot(alphas, coef[:,i],'r--',lw=2, label = '$u_xu_{2x}$') #-5.0e-07uxu2x
    elif i== lam3:
        ax.plot(alphas, coef[:,i],'g--',lw=2, label = '$uu_{3x}$')   #-2.5e-07uu3x
    elif i== lam4:
        ax.plot(alphas, coef[:,i],'c--',lw=2, label = '$u^2u_{2x}$') # 1.25e-05u2u2x
    elif i== lam5:
        ax.plot(alphas, coef[:,i],'y--',lw=2, label = '$u_{4x}$')    #1.25e-09u4x
    elif i== lam6:
        ax.plot(alphas, coef[:,i],'k--',lw=2, label = '$uu_{2x}$')   #-1.25e-05uu2x

    else:
        ax.plot(alphas, coef[:,i], lw=2)    
#        pass  
#       
ax.set_xscale('log')
ax.set_xlim(np.max(alphas),np.min(alphas))  # reverse axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('alpha', size = 15,labelpad=0.2)
plt.ylabel('Coefficients', size = 15,labelpad=0.2)
plt.legend(loc=1)
plt.show()

#%%
#np.savez('coeff_data/mod_burS_str0.npz', coef=cf, dtol= otol, lam=alphas, test_mse=test_mse)