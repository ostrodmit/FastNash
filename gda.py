#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import math

from prox import prox

def gda(Gx,Gy,dx,dy,Rx,Ry,T,K,gam_x,gam_y,x0,y0):
    x = np.zeros([dx,T+1])
    x[:,0] = x0
    x_best = np.zeros([dx,T+1])
    x_best[:,0] = x0
    y = np.zeros([dy,T+1])
    y[:,0] = y0
    y_best = np.zeros([dy,T+1])
    y_best[:,0] = y0
    Gx_norm = math.inf * np.ones(T+1)
    Gx_norm_best = math.inf * np.ones(T+1)
    Gy_norm = math.inf * np.ones(T+1)
    for t in range(T):
        print(t)
        y_tmp = y[:,t]
        # Ty gradient ascent steps
        for k in range(K):
            gy_tmp = gam_y * Gy(x[:,t],y_tmp)
            y_tmp = prox(y_tmp,-gy_tmp,Ry)
        y[:,t+1] = y_tmp
        gxt = gam_x * Gx(x[:,t],y_tmp)
        x[:,t+1] = prox(x[:,t],gxt,Rx)
        Gx_norm[t] = la.norm(Gx(x[:,t],y[:,t]))
        Gy_norm[t] = la.norm(Gy(x[:,t],y[:,t]))
        print(Gx_norm[t])
        Gx_norm_list = Gx_norm.tolist()
        tau = Gx_norm_list.index(min(Gx_norm))
        x_best[:,t] = x[:,tau]
        y_best[:,t] = y[:,tau]
        Gx_norm_best[t] = Gx_norm[tau]
    return x, y, Gx_norm, Gy_norm, x_best, y_best, Gx_norm_best