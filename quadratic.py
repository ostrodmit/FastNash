#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la

def quad_grad(A,b,x):
    r = np.matmul(A,x)-b
    g = np.matmul(A.T,r)
    return(g)

def quad_func(A,b,x):
    f = ((la.norm(np.matmul(A,x)-b))**2)/2
    return(f)