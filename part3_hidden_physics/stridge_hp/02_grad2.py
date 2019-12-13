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


## SPARSE REGRESSION ##############

def AIC(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
    m = testy.shape[0]
    aic = m*np.log(sse/m) + 2*k
    
    return aic
   
def BIC(testy,  testr, we, k):
    res = testy - testr.dot(we.real)
    sse = np.sum(res**2)
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
            pred = TestR.dot(w.real)
            test_mse =  mean_squared_error(TestY,  TestR.dot(w.real))  
            cmplx = np.count_nonzero(w)
#            
            aicc  = AICc(TestY, TestR, w, cmplx)
            aic   = AIC(TestY, TestR, w, cmplx)
            bic   = BIC(TestY, TestR, w, cmplx)
            bicc  = BICc(TestY, TestR, w, cmplx)

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal Parameters:", cmplx, test_mse, tol_best)
     
    
    return  w_best.real, pred, tol_best, test_mse, cmplx,  aicc, aic, bic, bicc

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
print(os.listdir("./../data_gen/"))
path = './../data_gen/train_data/'

data = np.load(path+'02_train_grad2.npy')
data_ref = pd.read_csv(path+"02_train_grad2.csv",  encoding='utf-8')  

#%%
nf = 10
thetad = data[:,:-1]
theta  = thetad[::nf,:]

pid    = data[:,-1].reshape((-1,1))
pi     = pid[::nf, :] 

#%%

n_alphas = 50 #200
alphas = np.logspace(-8, 2, n_alphas) #-16 14 -8 2
dtol = 100 #10 200

otol     = np.empty(n_alphas)
test_mse = np.empty(n_alphas)
aicc     = np.empty(n_alphas)
aic      = np.empty(n_alphas)
bic      = np.empty(n_alphas)
bicc     = np.empty(n_alphas)
aicc     = np.empty(n_alphas)
cmplx    = np.empty(n_alphas)

coefs = []
pred  = []
for i, a in enumerate(alphas):
    w, pe, otol[i], test_mse[i], cmplx[i], aicc[i],  aic[i],  bic[i],  bicc[i] = TrainSTRidge(theta,pi,a,dtol)
    coefs.append(w)
    pred.append(pe)    
    
#%%    
coef = np.array(coefs)
pred = np.array(pred)

cf  = coef.reshape(coef.shape[0],coef.shape[1])
ped = pred.reshape(pred.shape[0],pred.shape[1])

param = np.stack((cmplx, aicc, aic, bic, bicc, test_mse, otol, alphas), axis=1)
temp1 = np.column_stack((param, cf))
tot   = np.column_stack((temp1, ped))

opt_param   = tot[tot[:,1].argsort()]
model       = opt_param[:,temp1.shape[1]:].T
model_coef  = opt_param[:,param.shape[1]:temp1.shape[1]]
model_coefT = opt_param[:,param.shape[1]:temp1.shape[1]].T


#%%Psot processing
pdf = 'results/coeff_grad2b1.pdf'        
eps = 'results/coeff_grad2b1.eps' 

ax = plt.gca()
for i in range(coef.shape[1]):
    ax.plot(alphas, coef[:,i].real, lw=2)    
       
ax.set_xscale('log')
ax.set_xlim(np.max(alphas),np.min(alphas))  # reverse axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel(r'$\lambda$', size = 15,labelpad=0.2)
plt.ylabel('Coefficients', size = 15,labelpad=0.2)
#plt.legend(loc=1)
plt.savefig(pdf)
plt.savefig(eps)
plt.show()


#%%

def print_model(model_coef_, data_ref_, s_):
 
    idx = np.nonzero(model_coef_[s_,:])
    mid = model_coef[s_,idx][0]
    col = np.array(list(data_ref_.columns.values))
    col_mid = col[idx]
    
    t = 'model:' + ' '
    for m , ix in enumerate(col_mid):
            temp = str(np.around(mid[m], decimals=14)) + ix
            t =  t + ' ' + temp
    print(t)
    
select = 80
print_model(model_coef, data_ref, select)


path_2= './../data_gen/test_data/'
data_test = np.load(path_2+'02_test_grad2.npy')
theta_test = data_test[:,:-1]
pi_test   = data_test[:,-1].reshape((-1,1))
pred_model = theta_test.dot(model_coefT[:,select]) 

num_bins = 64

fig, axs = plt.subplots(1,1,figsize=(9,5))

# the histogram of the data
ntrue, binst, patchest = axs.hist(pi_test.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,linewidth=2.0,range=(-4*np.std(pi),4*np.std(pi)),density=True,
                                 label="True")

#ntrue, binst, patchest = axs.hist(pi.flatten(), num_bins, histtype='step', alpha=1, color='g',zorder=5,linewidth=2.0,range=(-4*np.std(t11),4*np.std(t11)),density=True,
#                                 label="Samg")

ntrue, binst, patchest = axs.hist(pred_model.flatten(), num_bins, histtype='step', alpha=1, color='b',zorder=5,linewidth=2.0,range=(-4*np.std(pi),4*np.std(pi)),density=True,
                                 label="STRidge")


x_ticks = np.arange(-4*np.std(pi), 4.1*np.std(pi), np.std(pi))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]

axs.set_title(r"$\pi$")
axs.set_yscale('log')
axs.set_xticks(x_ticks)                              


# Tweak spacing to prevent clipping of ylabel
axs.legend()  
fig.tight_layout()
plt.show()
#fig.savefig('results/Pi_grad02b1.eps')
#fig.savefig('results/Pi_grad02b1.pdf')
#%%
base =  np.array(['wxx', 'wyy',  
                 '|s|', 'dw', '|w|', '|k|'])
cand = len(base)
for i in range(cand):
    for j in range(i, cand):
#        print(j)
        th = base[i] + base[j]        
        base= np.hstack((base, th))        
        
columns = ['# terms', 'aicc', 'aic', 'bic', 'bicc', 'test_mse', 'otol','alpha','c']  
ebase    = base.tolist()
columns = columns + ebase

lib= pd.DataFrame(opt_param[:,:temp1.shape[1]], columns=columns)

#%%
#lib.to_csv('coeff_data/grad02b1_coeff.csv',index=None)
#np.savez('coeff_data/grad02b1_str.npz', coef=cf, pred = model, dtol= otol, lam=alphas, test_mse=test_mse, aicc = aicc, aic=aic, bic=bic, bicc=bicc)