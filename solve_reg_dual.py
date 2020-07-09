#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy as np
from restart_fgm import restart_fgm

def solve_reg_dual(y,x_prv,y_bar,gam_x,lam_y,T,S,Rx,Gx,Gy):
    g_reg_x = lambda x: Gx(x,y) - 1/gam_x * (x-x_prv)
    x_nxt = restart_fgm(x_prv,Rx,2/3*gam_x,T,S,g_reg_x)
    g_reg_y = -Gy(x_nxt,y) + lam_y * (y-y_bar)
    return x_nxt, g_reg_y