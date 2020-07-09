#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import math

from restart_fgm import restart_fgm
from solve_reg_dual import solve_reg_dual

def fast_nash(Gx,Gy,dx,dy,Rx,Ry,x0,y_bar,Tx,Ty,Sy,gam_x,gam_y,lam_y,To,So):
    x = np.zeros([dx,Tx+1])
    x_best = np.zeros([dx,Tx+1])
    x[:,0] = x0
    x_best[:,0] = x0
    Gx_norm = math.inf * np.ones(Tx+1)
    print(Gx_norm.shape)
    Gx_norm_best = Gx_norm
    y = np.zeros([dy,Ty])
    y_best = np.zeros([dy,Ty])
    for t in range(1,Tx+1):
        print(t)
        xt_y = \
        lambda y: solve_reg_dual(y,x[:,t-1],y_bar,gam_x,lam_y,To,So,Rx,Gx,Gy)[0]
        gt_reg_y = \
        lambda y: solve_reg_dual(y,x[:,t-1],y_bar,gam_x,lam_y,To,So,Rx,Gx,Gy)[1]
        print('here1')
        yy = restart_fgm(y_bar,Ry,gam_y,Ty,Sy,gt_reg_y)[0]
        print('here2')
        y[:,t] = yy[:,Sy]
        print('here3')
        x[:,t] = xt_y(y[:,t])
        print('here4')
        Gx_norm[t] = la.norm(Gx(x[:,t],y[:,t]))
        print('here5')
        Gx_norm_list = Gx_norm.tolist()
        tau = Gx_norm_list.index(min(Gx_norm))
        print('here6')
        x_best[:,t] = x[:,tau]
        y_best[:,t] = y[:,tau]
        Gx_norm_best[t] = Gx_norm[tau]
    return x_best, y_best, Gx_norm_best