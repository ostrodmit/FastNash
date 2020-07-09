#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from fast_nash import noncvx_ccv

# Testing noncvx_ccv on the primal-dual formulation of the regression problem 
# min_x \|\tanh(x) - b\|_2 where 
