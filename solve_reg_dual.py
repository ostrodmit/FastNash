#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from restart_FGM import restart_FGM

def solve_reg_dual(y,x_prev,y_bar,gam_x,lam_y,T,S,R):
    x_new = restart_FGN(x_prev,R,2*gam_x/3,T,S,???)