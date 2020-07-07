#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from FGM import FGM

def quad_grad(A,b,x):
    r = np.matmul(A,x) - b
    g = np.matmul(A.T,r)
    return(g)

def quad_func(A,b,x):
    f = ( (la.norm(np.matmul(A,x) - b)) ** 2 )/2
    return(f)

# Testing FGM on the unconstrained quadratic problem min_x (Ax-b)^2/2 
d = 30
T = 10**3;
A = np.diag(range(1,d+1))
b = np.ones([d,1])
# Stepsize calculation 
B = np.multiply(A.transpose(), A)
[l,v] = la.eigh(B)
L = np.max(l)
gam = 1/L

# Defining gradient oracle
grad = lambda x: quad_grad(A,b,x)

z0 = np.zeros(d)
opt_sol = la.solve(A,b)
print(opt_sol)
R = 2*la.norm(opt_sol)
z = FGM(z0,R,gam,T,grad)

# Plotting
f = np.zeros(T+1)
func = lambda x: quad_func(A,b,x)
for t in range(T+1):
    zt = z[:,[t]]
    f[t] = func(zt)
rate = [L * (R**2) * (t+1)**(-2) for t in range(T+1)]
plt.plot(f,color='red')
plt.plot(rate,color='grey')
plt.xscale('log')
plt.yscale('log')
plt.show()