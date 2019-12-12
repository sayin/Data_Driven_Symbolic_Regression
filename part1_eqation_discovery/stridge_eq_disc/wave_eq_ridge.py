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

#doublecheck the data is there
print(os.listdir("./../data_gen/clean/data/."))

# read in the data to pandas
data0 = pd.read_csv("./../data_gen/clean/data/wave_data.csv",  encoding='utf-8')    

data1 = np.array(data0)
theta = data1[:,:-1]
ut = data1[:,-1:]

u          = theta[:,0]
ux         = theta[:,1]
u2x        = theta[:,2]
u3x        = theta[:,3]
u4x        = theta[:,4]
u5x        = theta[:,5]   

c = np.ones((theta.shape[0],1))
theta = np.column_stack((c,theta))
#%%#%%
##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 50, STR_iters = 50,
                 l0_penalty = None, normalize = 2, 
                 split = 0.8, print_best_tol = False):
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

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
#    print d_tol
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=-1)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + \
                                        l0_penalty*np.count_nonzero(w_best)
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

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return  w_best

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = True):
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
def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y,rcond=-1)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
#%%
key= '''
#Using STRidge to predict the PDE uer = -1.0*ux  
'''
#n_alphas = 50
#alphas = np.logspace(-20, -2, n_alphas)

w = TrainSTRidge(theta,ut,10**-1, 5)
#coefs.append(w)+
  
#coeff = np.asarray(coefs)
# Display results
#ax = plt.gca()
#
#ax.plot(alphas, coeff)
#ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
#plt.xlabel('alpha')
#plt.ylabel('weights')
#plt.title('Ridge coefficients as a function of the regularization')
#plt.axis('tight')
#plt.show()

#%%
#from sklearn.linear_model import Ridge, Lasso, RidgeCV
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import scale 
#from sklearn.model_selection import train_test_split
#
#X_train, X_test , y_train, y_test = train_test_split(theta, ut, 
#                            test_size=0.2, random_state=1)
   
# #############################################################################
# Compute paths

#n_alphas = 100
#alphas = np.logspace(-50, -5, n_alphas)
##Xt = np.delete(X_train, 1, axis=1)
#coefs = []
#for a in alphas:
#    ridge = Ridge(alpha=a,max_iter=1000, fit_intercept=False)
#    ridge.fit(X_train, y_train.reshape(-1,))
#    coefs.append(ridge.coef_)    
##############################################################################
 
#clf = RidgeCV(alphas).fit(theta, ut)
#lam  = clf.alpha_
#sc  = clf.score(theta, ut)

#%% Display results
#file = 'results/wave_rcf.pdf'
#
#coef = np.array(coefs)
#lam1 = 2
#ax = plt.gca()
#for i in range(coef.shape[1]):
#    if i ==lam1:
#        ax.plot(alphas, coef[:,i],'b--',lw=2)
#    else:
#        ax.plot(alphas, coef[:,i], lw=2)      
#        
#ax.set_xscale('log')
#ax.set_xlim(np.max(alphas),np.min(alphas))  # reverse axis
#plt.xlabel('alpha', size = 15,labelpad=0.2)
#plt.ylabel('Coefficients', size = 15,labelpad=0.2)
###plt.title('Ridge coefficients as a function of the regularization')
#ax.grid(which='minor', alpha=0.2)
#ax.grid(which='major', alpha=0.5)
#plt.savefig(file)
#plt.show()