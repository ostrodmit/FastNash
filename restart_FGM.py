#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from FGM import FGM

def restart_FGM(z0,R,gam,T,S,grad_orc):
    d = len(z0)
    z_rx = np.zeros([d,S+1])
    z_new = np.zeros([d,T+1])
    z_all = np.zeros([d,(S+1)*(T+1)])
    z_rx[:,0] = z0
    for s in range(S):
        z_new = FGM(z_rx[:,s],R,gam,T,grad_orc)
        z_all[:,(T+1)*s:(T+1)*(s+1)] = z_new
        z_rx[:,s+1] = z_new[:,T]
    return z_rx, z_all