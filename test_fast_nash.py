#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import os
from plots import plots

from fast_nash import fast_nash
from fast_gda import fast_gda
from quadratic import quad_grad, quad_func

# Testing FastNash vs. FastGDA on the nonconvex-strongly-concave problem
# 1/2 * min_x max_y |tanh(A*x) - J*y|^2 - |J*y - ones|^2 - lam*(y[0]-1)^2
# A discrete derivative matrix w/o 1st element, J excludes the first element
# with L > 1 and lam << 1

d = 10
kap_y = 1000
Tx = int(2)

# discrete difference matrix [[0,0,...,0], [-1,1,0,...,0],..., [0,...,0,-1,1]]
I = np.identity(d)
A = np.identity(d)
A[0,0] = 0
for j in range(1,d):
    A[j,j-1] = -1
b = np.ones(d)

# matrix excluding the first element
E = np.identity(d)
E[0,0] = 0

# Jacombian of elementwise tanh(x)
def Jac(u):
    v = np.zeros(len(u))
    for k in range(len(u)):
        v[k] = np.cosh(u[k])**(-2)
    return np.diagflat(v)

# problem class parameters
B = np.matmul(A.T,A)
[l,v] = la.eigh(B)
#print(l)
Lxx = np.max(l)
#Lyy = kap_y
Lyy = 2*np.max(l)
Lxy = np.sqrt(Lxx)
#Lxy = 1
Lyy_plus = Lyy + Lxy ** 2/Lxx
lam_y = Lyy/kap_y
#lam_y = 0
#Ry=2*la.norm(b)
#Gap = lam_y * la.norm(b)**2/2
Gap = lam_y/2

x0 = np.zeros(d)
y_bar = np.zeros(d)
#x_opt = la.lstsq(A,b,rcond=None)[0]

# Defining oracle -- old problem
#Gx = lambda x,y: quad_grad(A,y,x)
#def Gy(x,y):
#    z = np.matmul(A,x)
#    #return quad_grad(I,z,y) - lam_y * quad_grad(I,b,y)
#    return quad_grad(I,z,y) - Lyy * quad_grad(I,b,y)
##func = lambda x,y: quad_func(A,y,x) - lam_y * quad_func(I,b,y)
#func = lambda x,y: quad_func(A,y,x) - Lyy * quad_func(I,b,y)

# Defining oracle -- new problem
#Gx = lambda x,y: np.matmul(J(x).T, quad_grad(A,np.matmul(E,y),np.tanh(x)))
def Gx(x,y):
    z = A.dot(x)
    r = np.tanh(z) - E.dot(y)
    jac = np.ones(len(z)) - np.tanh(z)**2
    rho = jac * r
    AT = A.T
    return AT.dot(rho)
def Gy(x,y):
    g = quad_grad(E,np.tanh(A.dot(x)),y) \
        -2*quad_grad(E,np.ones(d),y)  #-2*quad_grad(A,np.zeros(d),y)
    g[0] = -lam_y*(y[0]-1)
    return g
#func = lambda x,y: quad_func(A,y,x) - lam_y * quad_func(I,b,y)
func = lambda x,y: quad_func(I,E.dot(y),np.tanh(A.dot(x))) \
                    - 2*quad_func(E,np.ones(d),y) \
                    - lam_y * (y[0]-1)**2/2  # - 2*quad_func(A,np.zeros(d),y)

# Exact optimum from the optimality condition
#GMat \
#= np.block([[B,                  -np.matmul(A.T,J)],\
#            [-np.matmul(J.T,A), np.matmul(J.T,J) - 2*np.matmul(A.T,A) - lam_y*(I-J)]])
#GOff = np.zeros(2*d)
#GOff[d:2*d] = -lam_y*np.matmul(I-J,np.ones(d))
#z_opt = la.lstsq(GMat,GOff,rcond=None)[0]
#x_opt = z_opt[0:d]
#y_opt = z_opt[d:2*d]

#u_opt = la.lstsq(A,np.matmul(E,b),rcond=None)[0]
#x_opt = np.arctanh(u_opt)
x_opt = la.lstsq(A,E.dot(b),rcond=None)[0] # tanh disappears at optimum
y_opt = np.ones(d)
Rx = 2*la.norm(x_opt)
Ry = 2*la.norm(y_opt)
opt_val = func(x_opt,y_opt)


# Initializing input parameters for fast_nash
gam_x = 1/(2*Lxx)
gam_y = 1/(Lyy_plus)
# Ty = int(np.sqrt(40*(kap_y+1))) # conservative estimate
Ty = int(np.sqrt(4*(kap_y+1))) 
#Ty = 1
Theta = Lyy*(Ry**2)
Theta_plus = Lyy_plus*(Ry**2)
delta = 1e-2
Sy = int(np.ceil(2*np.log2(max([Ty,Theta_plus/delta]))))
#Sy = 1

# To = 11 # Conservative estimate
To = 4

# OverGap = 72*(3*Gap+2*Theta+6*lam_y*(Ry**2)) # Conservative estimate
OverGap = Gap+Theta+lam_y*(Ry**2)
So \
= int(np.ceil(np.log2(OverGap * (Tx/Gap + 2*Theta_plus/(delta**2) + 1/(12*delta)))/2))

# Running FastNash (without regularization)
print('')
print('Running FastNash')
x_nash, y_nash, Gx_norm_nash, Gy_norm_nash \
= fast_nash(Gx,Gy,d,d,Rx,Ry,x0,y_bar,Tx,Ty,Sy,gam_x,gam_y,0,To,So)
#, x_best_nash, y_best_nash, Gx_norm_best_nash

# Initializing missing input parameters for FastGDA
#K = 2*kap_y
#y0 = np.zeros(d)
gam_x_gda = 1/2/max(Lxx,Lxy**2/lam_y) # lam_y is 1
gam_y_gda = 1/Lyy

# Running FastGDA
print('')
print('Running FastGDA')
x_gda, y_gda, Gx_norm_gda, Gy_norm_gda \
= fast_gda(Gx,Gy,d,d,Rx,Ry,x0,y_bar,Tx,Ty,Sy,gam_x_gda,gam_y_gda,0)
#x_best_gda, y_best_gda, Gx_norm_best_gda

# Computing stats
Gx_norm_rate = [np.sqrt(10*Lxx*(Gap+2*lam_y*(Ry**2))/(t+1)) for t in range(Tx+1)]
nash_stretch = Ty * Sy * To * So
gda_stretch = Ty * Sy
nash_calls = [nash_stretch * t for t in range(Tx+1)]
gda_calls = [gda_stretch * t for t in range(Tx+1)]

F_nash = np.zeros(Tx+1)
F_gda = np.zeros(Tx+1)
for t in range(1,Tx+1):
    F_nash[t] = func(x_nash[:,t],y_nash[:,t])
    F_gda[t] = func(x_gda[:,t],y_gda[:,t])


# Saving results
fpath = os.getcwd()+'/data/'+'d-'+np.str(d)+'-kap-'+np.str(kap_y)+'-Tx-'+np.str(Tx)+'/'
os.makedirs(fpath,exist_ok=True)

np.savetxt(fpath+'nash-calls.txt', nash_calls)
np.savetxt(fpath+'gda-calls.txt', gda_calls)

np.savetxt(fpath+'nash-Gx_norm.txt', Gx_norm_nash)
np.savetxt(fpath+'gda-Gx_norm.txt', Gx_norm_gda)
np.savetxt(fpath+'rate-Gx_norm.txt', Gx_norm_rate)

np.savetxt(fpath+'nash-gap.txt', F_nash-opt_val)
np.savetxt(fpath+'gda-gap.txt', F_gda-opt_val)

plots(fpath)