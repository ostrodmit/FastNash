#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from FGM import FGM

def restart_FGM(z0,R,gam,T,S,grad_orc):
    d = len(z0)
    z_rx = np.zeros([d,S+1])
    z_all = np.zeros([d,S*(T+1)])
    z_rx[:,0] = z0
    for s in range(S):
        z_temp = FGM(z_rx[:,s],R,gam,T,grad_orc)
        z_rx[:,s+1] = z_temp[:,T]
        z_all[:,(T)*s:(T)*(s+1)] = z_temp[:,1:T+1]
    return z_rx, z_all