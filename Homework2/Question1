#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:49:38 2025

@author: pverma
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------ Define parameters -----------------------------------------
def default_pars(**kwargs):
    pars = {}
    pars['V_th'] = -54.     # spike threshold [mV]
    pars['V_reset'] = -80.  # reset potential [mV]
    pars['C_m'] = 10.       # membrane capacitance [nF/mm2]
    pars['g_L'] = 1.        # membrane conductance [muS/mm2]
    pars['g_ex'] = 0.       # initial excitatory conductance [muS/mm2 ]
    pars['V_init'] = -70.   # initial potential = E [mV]
    pars['E_L'] = -70.      # leak reversal potential [mV]
    pars['E_ex'] = 0        # excitatory external potential [mV]
    pars['tref'] = 0.       # refractory time (ms)
    pars['tau_ex'] = 10     # tau excitatory [ms]
    pars['delta_g'] = 0.5   # conductance inc after AP 

    # simulation parameters
    pars['T'] = 500.        # total duration (ms)
    pars['dt'] = 0.1        # time step (ms)
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # time array

    for k in kwargs:
        pars[k] = kwargs[k]

    return pars

# ------------------------Getting g_ex-----------------------------------------

def g_ex(pars, AP):
 
    g_init, tau_ex, delta_g = pars['g_ex'], pars['tau_ex'], pars['delta_g']
    dt, t = pars['dt'], pars['range_t']
    Lt = t.size
    
    g_ex = np.zeros(Lt)
    g_ex[0] = g_init
    
    for i in range(Lt - 1):
        dg = -g_ex[i]*dt / tau_ex
        
        if i in AP:
            dg = dg + delta_g
        
        g_ex[i+1] = g_ex[i] + dg
        
    return g_ex

#-----------------------Stimulating LIF ---------------------------------------

def run_LIF(pars, g_ex):
    V_th, V_reset = pars['V_th'], pars['V_reset']
    C_m, g_L = pars['C_m'], pars['g_L']
    V_init, E_L = pars['V_init'], pars['E_L']
    E_ex = pars['E_ex']
    dt, t = pars['dt'], pars['range_t']
    Lt = t.size

    # Initialize
    v = np.zeros(Lt)
    v[0] = V_init
    # Iinj = np.zeros(Lt)

    # # Inject current from 100 to 400 ms
    # Iinj[(t >= 100) & (t <= 400)] = Iinj_amp

    # spks = []

    for i in range(Lt - 1):
        if v[i] >= V_th:
            # spks.append(t[i])
            v[i-1] = -20
            v[i] = V_reset

        dv = (dt / C_m) * (-g_L * (v[i] - E_L) -g_ex[i] * (v[i] - E_ex))
        v[i+1] = v[i] + dv

    return v # np.array(spks)


#----------------------Stimulating I_ex----------------------------------------

def I_ex(pars, v, g_ex):
    E_ex = pars['E_ex']
    t = pars['range_t']
    Lt = t.size
    
    I_ex = np.zeros(Lt) # initialising excitatory I

    for i in range(Lt):
        I_ex[i] = g_ex[i] * (v[i] - E_ex)
          
    return I_ex        


# ------------------------Getting values for plotting--------------------------


AP = (1000, 2000, 2300, 3000, 3200, 4000, 4100) # AP/dt to get time indices
pars = default_pars() # Getting default params
g_ex = g_ex(pars, AP) # Producing g_ex values
v = run_LIF(pars, g_ex) # producting V values for the excitatory inputs 
I_ex = I_ex(pars, v, g_ex) # excitatory current 



#----------------------- Plotting V(t) and I_ex -------------------------------

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(pars['range_t'], v, 'k')
ax[0].axhline(y=-54, color='r', linestyle='--')
ax[0].set_ylabel('V (mV)')
ax[0].set_ylim([-90, 10])


ax[1].plot(pars['range_t'], I_ex, 'k')
ax[1].set_ylabel('I ex (nA)')
ax[1].set_xlabel('time (ms)')
ax[1].set_ylim([-55, 5])

