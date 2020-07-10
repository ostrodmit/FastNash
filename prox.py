#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.linalg as la

def prox(z,zeta,R):
    z_temp = z - zeta 
    norm = la.norm(z_temp)
    if norm > R:
        z_new = z_temp/norm*R
    else:
        z_new = z_temp
    #z_new = z_temp
    return z_new