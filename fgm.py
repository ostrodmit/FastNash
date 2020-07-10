#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from prox import prox

def fgm(z0,R,gam,T,grad_orc):
    d = len(z0)
    u = np.zeros([d,T])
    v = np.zeros([d,T+1])
    w = np.zeros([d,T+1])
    z = np.zeros([d,T+1])
    G = np.zeros([d,T+1])
    g = np.zeros([d,T])
    tau = np.zeros(T)
    z[:,0] = z0
    for t in range(T):
        u[:,t] = prox(z0, gam * G[:,t], R)
        tau[t] = 2*(t+2)/(t+1)/(t+4)
        v[:,t+1] = tau[t] * u[:,t] + (1-tau[t]) * z[:,t]
        # print(grad_orc(v[:,[t+1]]).shape)
        #g[:,[t]] = (t+2)/2 * grad_orc(v[:,[t+1]])
        g[:,t] = (t+2)/2 * grad_orc(v[:,t+1])
        w[:,t+1] = prox(u[:,t], gam * g[:,t], R)
        z[:,t+1] = tau[t] * w[:,t+1] + (1-tau[t]) * z[:,t]
        G[:,t+1] = G[:,t] + g[:,t]
    return(z)