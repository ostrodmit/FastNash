#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from fast_nash import fast_nash
from quadratic import quad_grad, quad_func

# Testing fast_nash and competitors on the nonconvex-strongly-concave problem
# min_x \max_y |A*x - y|_2^2/2 - L*|y - b|^2/2
# with b the all-ones vector and A the discrete derivative matrix

d = 1
kap_y = 4
delta = 1e-2
Tx = int(10)

# discrete difference matrix [[1,0,...,0], [-1,1,0,...,0],..., [0,...,0,-1,1]]
I = np.identity(d)
A = I
#for j in range(1,d):
#    A[j,j-1] = -1
b = np.ones(d)

# problem class parameters
B = np.matmul(A.T,A)
[l,v] = la.eigh(B)
Lxx = np.max(l)
Lyy = kap_y
#Lxy = np.sqrt(Lyy)
Lxy = 1
Lyy_plus = Lyy + Lxy**2/Lxx
#lam_y = Lyy_plus/kap_y
#lam_y = 0
Ry=2*la.norm(b)
#Gap = lam_y * la.norm(b)**2/2
Gap = Lyy * la.norm(b)**2/2

x0 = np.zeros(d)
y_bar = np.zeros(d)
x_opt = la.lstsq(A,b,rcond=None)[0]
Rx = 2*la.norm(x_opt)

# Defining oracle
Gx = lambda x,y: quad_grad(A,y,x)
def Gy(x,y):
    z = np.matmul(A,x)
    #return quad_grad(I,z,y) - lam_y * quad_grad(I,b,y)
    return quad_grad(I,z,y) - Lyy * quad_grad(I,b,y)
#func = lambda x,y: quad_func(A,y,x) - lam_y * quad_func(I,b,y)
func = lambda x,y: quad_func(A,y,x) - Lyy * quad_func(I,b,y)

# Initializing input parameters for fast_nash
gam_x = 1/(2*Lxx)
gam_y = 1/(Lyy_plus)
Ty = int(np.sqrt(40*(kap_y+1)))
#Ty = 1
Theta_plus = Lyy_plus*(Ry**2)
Sy = int(np.ceil(2*np.log2(max([Ty,Theta_plus/delta]))))
#Sy = 1
To = 11
#OverGap = 72*(3*Gap+2*Theta_plus+6*lam_y*(Ry**2))
OverGap = 72*(3*Gap+8*Theta_plus)
So \
= int(np.ceil(np.log2(OverGap * (Tx/Gap + 2*Theta_plus/(delta**2) + 1/(12*delta)))/2))


# Running fast_nash (without regularization)
x, y, Gx_norm, Gy_norm, x_best, y_best, Gx_norm_best  \
= fast_nash(Gx,Gy,d,d,Rx,Ry,x0,y_bar,Tx,Ty,Sy,gam_x,gam_y,0,To,So)

# Plotting
plt.plot(Gx_norm,color='red')
plt.plot(Gy_norm,color='green')
plt.xscale('log'); plt.yscale('log')
plt.show()

F = np.zeros(Tx+1)
for t in range(Tx+1):
    F[t] = func(x[:,t],y[:,t])
#rate = [L*(R**2)*(t+1)**(-2) for t in range(T+1)]
plt.plot(F,color='blue')
plt.xscale('log'); plt.yscale('log')
plt.show()