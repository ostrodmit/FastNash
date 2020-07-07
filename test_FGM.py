#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from FGM import FGM
from restart_FGM import restart_FGM

def quad_grad(A,b,x):
    r = np.matmul(A,x)-b
    g = np.matmul(A.T,r)
    return(g)

def quad_func(A,b,x):
    f = ((la.norm(np.matmul(A,x)-b))**2)/2
    return(f)

# Testing FGM on the unconstrained quadratic problem min_x (Ax-b)^2/2 
d = 50
T = 10**3;
A = np.diag(range(d))
b = np.ones([d,1])
b[0] = 0
# Stepsize calculation
B = np.multiply(A.transpose(), A)
[l,v] = la.eigh(B)
L = np.max(l)
gam = 1/L

# Defining gradient oracle
grad = lambda x: quad_grad(A,b,x)

z0 = np.zeros(d)
opt_sol = la.lstsq(A,b)[0]
R = 2*la.norm(opt_sol)
z = FGM(z0,R,gam,T,grad)

# Plotting
func = lambda x: quad_func(A,b,x)
f = np.zeros(T+1)
for t in range(T+1):
    f[t] = func(z[:,[t]])
rate = [L*(R**2)*(t+1)**(-2) for t in range(T+1)]
plt.plot(f,color='red')
plt.plot(rate,color='grey')
plt.xscale('log'); plt.yscale('log')
plt.show()

# Testing restarted FGM on the regularized problem min_x (Ax-b)^2/2 + mu*x^2/2
eps = 1e-2
kappa = 1e4
mu = L/kappa
def grad_reg(x):
    g = grad(x) + mu * x
    return g
I = np.identity(d)
opt_sol_reg = la.solve(A+mu*I,b)
T_rx = int(np.ceil(np.sqrt(40*kappa)))
S = int(np.ceil(np.log2(3*L*R/eps)))
z_rx, z_all = restart_FGM(z0,R,gam,T_rx,S,grad_reg)

# Plotting
def func_reg(x):
    return func(x) + mu*(la.norm(x)**2)/2
opt_val_reg = func_reg(opt_sol_reg)
gap_reg_all = np.zeros((T_rx+1)*(S+1))
for t in range((T_rx+1)*(S+1)):
    gap_reg_all[t] = func_reg(z_all[:,t]) - opt_val_reg
#rate_rx = [L*(R**2)*(t+1)**(-2) for t in range(S+1)]
plt.plot(gap_reg_all,color='blue')
#plt.plot(rate,color='grey')
#plt.xscale('log')
plt.yscale('log')
plt.show()