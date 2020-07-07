#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

def FGM(z0,R,gam,T,grad_orc):
    d = len(z0)
    u = np.zeros([d,T])
    v = np.zeros([d,T+1])
    w = np.zeros([d,T+1])
    z = np.zeros([d,T+1])
    G = np.zeros([d,T+1])
    g = np.zeros([d,T])
    tau = np.zeros(T)
    for t in range(T):
        u[:,t] = prox(z0, gam * G[:,t], R)
        tau[t] = 2*(t+2)/(t+1)/(t+4)
        v[:,t+1] = tau[t] * u[:,t] + (1-tau[t]) * z[:,t]
        g[:,[t]] = (t+2)/2 * grad_orc(v[:,[t+1]])
        w[:,t+1] = prox(u[:,t], gam * g[:,t], R)
        z[:,t+1] = tau[t] * w[:,t+1] + (1-tau[t]) * z[:,t]
        G[:,t+1] = G[:,t] + g[:,t]
    return(z);

def prox(z,zeta,R):
    z_temp = z - zeta 
    norm = la.norm(z_temp)
    if norm > R:
        z_new = z_temp/norm
    else:
        z_new = z_temp
    return z_new