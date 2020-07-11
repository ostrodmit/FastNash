#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plots(fpath):
    # Loading data
    nash_calls = np.loadtxt(fpath+'nash-calls.txt')
    gda_calls = np.loadtxt(fpath+'gda-calls.txt')

    nash_Gx_norm = np.loadtxt(fpath+'nash-Gx_norm.txt')
    gda_Gx_norm = np.loadtxt(fpath+'gda-Gx_norm.txt')
    rate_Gx_norm = np.loadtxt(fpath+'rate-Gx_norm.txt')

    nash_gap = np.loadtxt(fpath+'nash-gap.txt')
    gda_gap = np.loadtxt(fpath+'gda-gap.txt')
    
    # Plotting Gx norm
    plt.plot(nash_calls[1:], nash_Gx_norm[1:],color='red',linewidth=2)
    plt.plot(gda_calls[1:],  gda_Gx_norm[1:],color='blue',linewidth=2)
    plt.plot(nash_calls[1:], rate_Gx_norm[1:],color='grey',linewidth=2)
    #plt.plot(Gy_norm,color='green')
    plt.xscale('log'); plt.yscale('log')
    plt.legend(['FastNash','FastGDA','Theory'])
    plt.title('x-gradient norm')
    plt.show()
    
    # Plotting objective gap
    #rate = [L*(R**2)*(t+1)**(-2) for t in range(T+1)]
    plt.plot(nash_calls[1:], nash_gap[1:], color='red',linewidth=2)
    plt.plot(gda_calls[1:],  gda_gap[1:], color='blue',linewidth=2)
    plt.xscale('log'); plt.yscale('log')
    plt.legend(['FastNash','FastGDA'])
    plt.title('objective gap')
    plt.show()