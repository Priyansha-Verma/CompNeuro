#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:12:51 2025

@author: pverma
"""

import numpy as np 
import matplotlib.pyplot as plt 

# ---------------------------Defining Parameters ------------------------------
def default_pars(**kwargs):
    pars = {}
    pars['V_th'] = -54.     # spike threshold [mV]
    pars['V_reset'] = -80.  # reset potential [mV]
    pars['C_m'] = 10.       # membrane capacitance [nF/mm2]
    pars['g_L'] = 1.        # leak conductance [uS/mm2]
    pars['g_ex'] = 0.       # initial excitatory conductance
    pars['V_init'] = -70.   # initial membrane potential = E_L
    pars['E_L'] = -70.      # leak reversal potential
    pars['E_ex'] = 0        # excitatory reversal potential
    pars['tref'] = 0.       # refractory time
    pars['tau_ex'] = 5      # tau_ex (ms)

    # STDP-related parameters
    pars['delta_US'] = 1.2          # Fixed change from US stimulus [uS/mm2]
    pars['delta_CS_initial'] = 0.0  # Initial value for delta_CS (can override)
    pars['A_LTP'] = 0.35
    pars['A_LTD'] = 0.4
    pars['tau_LTP'] = 25
    pars['tau_LTD'] = 35

    # Simulation time settings
    pars['T'] = 1000.        # Total time in ms (will update for run 2)
    pars['dt'] = 0.1         # Time step
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

    for k in kwargs:
        pars[k] = kwargs[k]

    return pars


# -------------------------------------------------

def stimulus_indices(pars, run_id):
    dt = pars['dt']
    
    if run_id == 1:
        us_times = [100, 200, 300, 400, 500, 600]
        cs_times = [90, 190, 290, 390, 490, 590, 690, 790, 890]
    elif run_id == 2:
        us_times = [100, 200, 300, 400, 500, 600]
        cs_times = [110, 210, 310, 410, 510, 610, 710, 810, 910]
    else:
        raise ValueError("run_id must be 1 or 2")

    us_idx = [int(t / dt) for t in us_times]
    cs_idx = [int(t / dt) for t in cs_times]

    return us_idx, cs_idx

# -------------------------
def run_LIF_with_STDP(pars, us_idx, cs_idx):
    # Unpack parameters
    dt, t = pars['dt'], pars['range_t']
    V_th, V_reset = pars['V_th'], pars['V_reset']
    C_m, g_L = pars['C_m'], pars['g_L']
    V_init, E_L = pars['V_init'], pars['E_L']
    E_ex = pars['E_ex']
    tau_ex = pars['tau_ex']
    
    A_LTP = pars['A_LTP']
    A_LTD = pars['A_LTD']
    tau_LTP = pars['tau_LTP']
    tau_LTD = pars['tau_LTD']
    
    delta_US = pars['delta_US']
    delta_CS = pars['delta_CS_initial']
    max_CS, min_CS = 1.2, 0.0

    # Initialize variables
    Lt = len(t)
    v = np.zeros(Lt)
    g_ex = np.zeros(Lt)
    spike_train = np.zeros(Lt)
    delta_CS_trace = np.zeros(Lt)

    v[0] = V_init
    last_post_spike_time = -np.inf
    last_cs_time = -np.inf

    for i in range(Lt - 1):
        delta_CS_trace[i] = delta_CS
        
        g_ex_current = g_ex[i]
        if i in us_idx:
            g_ex_current += delta_US
        if i in cs_idx:
            g_ex_current += delta_CS
            last_cs_time = i * dt
        
            # LTD
            delta_t = (i * dt) - last_post_spike_time
            if delta_t > 0:
                delta_CS -= A_LTD * np.exp(-delta_t / tau_LTD)
        
        # Decay g_ex using updated value
        g_ex[i + 1] = g_ex_current - (dt / tau_ex) * g_ex_current


        # Compute membrane potential
        if v[i] >= V_th:
            v[i] = V_reset
            spike_train[i] = 1
            last_post_spike_time = i * dt

            # LTP if a recent CS occurred
            delta_t = last_post_spike_time - last_cs_time
            if delta_t > 0:
                delta_CS += A_LTP * np.exp(-delta_t / tau_LTP)


        dv = (dt / C_m) * (-g_L * (v[i] - E_L) - g_ex[i] * (v[i] - E_ex))
        v[i + 1] = v[i] + dv

        # Clip delta_CS
        delta_CS = np.clip(delta_CS, min_CS, max_CS)

    return v, spike_train, g_ex, delta_CS_trace


# -----------Plotting function

def plot_results(pars, v, g_ex, delta_CS_trace, us_idx, cs_idx, title=""):
    t = pars['range_t']
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, v, label='Membrane Potential (v)')
    axs[0].axhline(pars['V_th'], color='gray', linestyle='--', label='V_th')
    axs[0].set_ylabel('Membrane\nPotential (mV)')
    axs[0].legend(loc='upper right')
    axs[0].set_title(title)

    axs[1].plot(t, g_ex, label='Excitatory Conductance (g_ex)', color='green')
    axs[1].set_ylabel('g_ex (uS/mm²)')
    axs[1].legend(loc='upper right')

    axs[2].plot(t, delta_CS_trace, label='Δg_CS over time', color='purple')
    axs[2].set_ylabel('Δg_CS (uS/mm²)')
    axs[2].set_xlabel('Time (ms)')
    axs[2].legend(loc='upper right')

    # Mark stimulus times
    for ax in axs:
        ax.vlines([t[i] for i in us_idx], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  color='red', linestyle='--', linewidth=0.8, label='US' if ax==axs[0] else "")
        ax.vlines([t[i] for i in cs_idx], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  color='blue', linestyle=':', linewidth=0.8, label='CS' if ax==axs[0] else "")
        if ax == axs[0]:
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# Run and plot for both Run 1 and Run 2
for run_id in [1, 2]:
    # Update delta_CS_initial based on run
    if run_id == 1:
        delta_CS_initial = 0.0
    elif run_id == 2:
        delta_CS_initial = 1.0

    # Define parameters and stimuli
    pars = default_pars(T=1000.0, delta_CS_initial=delta_CS_initial)
    us_idx, cs_idx = stimulus_indices(pars, run_id=run_id)

    # Run simulation
    v, spike_train, g_ex, delta_CS_trace = run_LIF_with_STDP(pars, us_idx, cs_idx)

    # Plot results
    plot_results(pars, v, g_ex, delta_CS_trace, us_idx, cs_idx, 
                 title=f"Run {run_id}: Classical Conditioning")

