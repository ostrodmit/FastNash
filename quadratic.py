#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
import numpy.linalg as la

def quad_grad(A,b,x):
    r = A.dot(x)-b
    A_T = A.T
    g = A_T.dot(r)
    return(g.T)

def quad_func(A,b,x):
    f = ((la.norm(A.dot(x)-b))**2)/2
    return(f)