#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:39:40 2025

@author: pverma
"""
import math
import numpy as np 
import matplotlib.pyplot as plt

# -------------------------Default Parameters----------------------------------
def default_pars(**kwargs):
    pars = {}
    pars['tau_m'] = 10.         # membrane time constant [ms]
    pars['V_init'] = -70.         # initial membrane potential 
    pars['V_th'] = -54.         # spike threshold [mV] 
    pars['V_reset'] = -80.      # reset potential [mV]
    pars['sigma_v'] = np.arange(0, 11, 0.5)    # noisy input (s.d. values 0->10)
    
    # stimulation parameters
    pars['T'] = 10000.           # total duration [ms] 
    pars['dt'] = 0.1            # time step [ms]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # time array
    
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars

# -----------------------Stimulating Noisy E_eff-------------------------------
def E_eff(pars):
    dt, t = pars['dt'], pars['range_t']
    sigma_v = pars['sigma_v']
    tau_m = pars['tau_m']
    L_sigma = sigma_v.size
    Lt = t.size
    
    e_eff = np.zeros((L_sigma,Lt))
    
    for i in range(L_sigma):
        e_eff[i, :] = -56 +  sigma_v[i] * math.sqrt(2* tau_m / dt) * np.random.randn(Lt)
            
    return e_eff      
# -------------------------Stimulating the model-------------------------------

def run_LIF(pars, E_eff):
    V_th, V_reset, V_init = pars['V_th'], pars['V_reset'], pars['V_init']
    sigma_v = pars['sigma_v']
    tau_m = pars['tau_m']
    dt, t = pars['dt'], pars['range_t']   
    L_sigma = sigma_v.size
    Lt = t.size
 
    # Initialize
    v = np.zeros((L_sigma, Lt))
    v[:, 0] = V_init
    spks = np.zeros((L_sigma, Lt))
 
    for row in range(L_sigma):
        for column in range(Lt-1):
            if v[row,column] >= V_th:
                spks[row, column] = 1
                v[row,column] = V_reset
 
            dv = (dt / tau_m) * (- v[row, column] + E_eff[row, column])
            v[row, column+1] = v[row, column] + dv
 
    return v , np.array(spks)




# =========================================================================== #

# -----------------------------First Part--------------------------------------
# Getting V for all sigmas for the first part with Vth = 1000mV for T=1000 ms

pars_NoSpikes = default_pars(V_th = 1e3, T=5000) # setting params
E_eff_NoSpikes = E_eff(pars_NoSpikes) # Producing random noise 
v_sigmaV, _ = run_LIF(pars_NoSpikes, E_eff_NoSpikes) # membrane potential matrix for all sigmas

    
# Calculating sd for each noise level     
sd = np.std(v_sigmaV, axis=1)

# Plotting s.d. in V v.s. sigma_v
plt.figure()
plt.plot(pars_NoSpikes['sigma_v'], sd)
plt.plot(pars_NoSpikes['sigma_v'], pars_NoSpikes['sigma_v'], color='grey', ls='--')
plt.gca().set_aspect('equal')
plt.title(f"Vth = {pars_NoSpikes['V_th']} and T = {pars_NoSpikes['T']}")
plt.xlabel('sigma_v')
plt.ylabel('s.d. V (mV)')


# ----------------------------Second Part--------------------------------------
# Stimulating noisy input model and computing membrane potential chnages

pars = default_pars()
E_eff = E_eff(pars)
v_sigma, spikes = run_LIF(pars, E_eff)

            
#Calculating spike average
av_spikes = np.sum(spikes, axis=1)*1000/pars['T'] # frequency in Hz

plt.figure()
plt.plot(pars['sigma_v'], av_spikes)
plt.xlabel('sigma_v')
plt.ylabel('Frequency (Hz)')



