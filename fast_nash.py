#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import math

from restart_fgm import restart_fgm
from solve_reg_dual import solve_reg_dual

def fast_nash(Gx,Gy,dx,dy,Rx,Ry,x0,y_bar,Tx,Ty,Sy,gam_x,gam_y,lam_y,To,So):
    x = np.zeros([dx,Tx+1])
    x[:,0] = x0
#    x_best = np.zeros([dx,Tx+1])
#    x_best[:,0] = x0
    y = np.zeros([dy,Tx+1])
    stretch = Ty*Sy
    x_all = np.zeros([dx,stretch*(Tx+1)])
    y_all = np.zeros([dy,stretch*(Tx+1)])
#    y_best = np.zeros([dy,Tx+1])
    # Formally computing y_0 to get the same intitial pair as other algorithms
    gt_reg_y = lambda y: -gam_y * Gy(x[:,0],y)
    yy_rx, yy_all = restart_fgm(y_bar,Ry,gam_y,Ty,Sy,gt_reg_y)
    y[:,0] = yy_rx[:,Sy]
    y_all[:,0:stretch] = yy_all
    x_all[:,0:stretch] = np.tile(x[:,0],[stretch,1]).T
    Gx_norm = math.inf * np.ones(Tx+1)
    Gx_norm[0] = la.norm(Gx(x[:,0],y[:,0]))
#    Gx_norm_best = math.inf * np.ones(Tx+1)
    Gy_norm = math.inf * np.ones(Tx+1)
    Gx_norm_all = math.inf * np.ones(stretch*(Tx+1))
    Gy_norm_all = math.inf * np.ones(stretch*(Tx+1))
    Gz_norm_all = math.inf * np.ones(stretch*(Tx+1))
    for t in range(1,Tx+1):
        print(str(t)+'/'+str(Tx))
        xt_y = \
        lambda y: solve_reg_dual(y,x[:,t-1],y_bar,gam_x,lam_y,To,So,Rx,Gx,Gy)[0]
        gt_reg_y = \
        lambda y: solve_reg_dual(y,x[:,t-1],y_bar,gam_x,lam_y,To,So,Rx,Gx,Gy)[1]
        yy_rx, yy_all = restart_fgm(y_bar,Ry,gam_y,Ty,Sy,gt_reg_y)
        y[:,t] = yy_rx[:,Sy]
        x[:,t] = xt_y(y[:,t])
        Gx_norm[t] = la.norm(Gx(x[:,t],y[:,t]))
        Gy_norm[t] = la.norm(Gy(x[:,t],y[:,t]))
        print(Gx_norm[t])
        #print(Gy_norm[t])
#        Gx_norm_list = Gx_norm.tolist()
#        tau = Gx_norm_list.index(min(Gx_norm))
#        x_best[:,t] = x[:,tau]
#        y_best[:,t] = y[:,tau]
#        Gx_norm_best[t] = Gx_norm[tau]
        ts = np.arange(t*stretch,(t+1)*stretch)
        y_all[:,ts] = yy_all
        x_all[:,ts] = np.tile(x[:,t],[stretch,1]).T
        for tau in ts:
            Gx_norm_all[tau] = la.norm(Gx(x_all[:,tau],y_all[:,tau]))
            Gy_norm_all[tau] = la.norm(Gy(x_all[:,tau],y_all[:,tau]))
            Gz_norm_all[tau] = Gx_norm_all[tau] + Gy_norm_all[tau]
    return x, y, Gx_norm, Gy_norm, Gz_norm_all#, x_best, y_best, Gx_norm_best