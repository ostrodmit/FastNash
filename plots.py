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
    
#    nash_Gz_norm_all = np.loadtxt(fpath+'nash-Gz_norm_all.txt')
#    gda_Gz_norm_all = np.loadtxt(fpath+'gda-Gz_norm_all.txt')

    nash_gap = np.loadtxt(fpath+'nash-gap.txt')
    gda_gap = np.loadtxt(fpath+'gda-gap.txt')
    
    # Plotting Gx norm
    plt.plot(nash_calls, nash_Gx_norm,color='red',linewidth=2)
    plt.plot(gda_calls,  gda_Gx_norm,color='blue',linewidth=2)
    plt.plot(nash_calls, rate_Gx_norm,color='grey',linewidth=2)
#    plt.xscale('log')
    plt.yscale('log')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(['FastNash','FastGDA','Theory'])
    plt.title('Primal gradient norm over primal iterations')
    plt.show()
        
    # Plotting total gradient norm in-between restarts
#    coeff = np.ceil(len(gda_Gz_norm_all)/len(nash_Gz_norm_all))
#    ts = [coeff * ts for ts in range(len(nash_Gz_norm_all))]
#    plt.plot(ts,nash_Gz_norm_all,color='red',linewidth=2)
#    plt.yscale('log')
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    plt.legend(['FastNash'])
#    plt.title('Full gradient norm over oracle calls (in two nested loops)')    
#    plt.show()
#    
#    plt.plot(gda_Gz_norm_all,color='blue',linewidth=2)
#    plt.yscale('log')
#    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    plt.legend(['FastGda'])
#    plt.title('Full gradient norm over oracle calls')
#    plt.show()

    
    # Plotting objective gap
    #rate = [L*(R**2)*(t+1)**(-2) for t in range(T+1)]
    plt.plot(nash_calls, nash_gap, color='red',linewidth=2)
    plt.plot(gda_calls,  gda_gap, color='blue',linewidth=2)
#    plt.xscale('log')
    plt.yscale('log')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(['FastNash','FastGDA'])
    plt.title('Objective gap over primal iterations')
    plt.show()