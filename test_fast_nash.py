#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from fast_nash import fast_nash

# Testing noncvx_ccv on the (unconstrained) nonconvex-strongly-concave problem
# min_x \max_y \|Ax - y\|_2^2 - \lam_y \|y - b\|^2 
# with b the all-ones vector and A the discrete derivative matrix

d = 50
Tx = int(1e4)
