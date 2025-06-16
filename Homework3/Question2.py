#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 00:13:57 2025

@author: pverma
"""

import numpy as np 
import matplotlib.pyplot as plt 


# ----------------------- Defining the Parameters -----------------------------
def default_pars(**kwargs):
    pars = {}
    # pars['rate'] = 10.              # Firing rate (Hz)
    pars['Po_dep'] = 1.             # Initial probablity depressing synapse
    pars['Pf_dep'] = 0.             # Final probability depressing synapse
    pars['tau_dep'] = 300.          # tau depressing synapse (ms)
    pars['Po_fac'] = 0.             # Initial Probablity facilitating synapse
    pars['Pf_fac'] = 1.
    pars['tau_fac'] = 100.          # tau facilitating synapse (ms)
    
    # Simulation time settings
    pars['T'] = 1000.        # Total time (ms) 
    pars['dt'] = 0.1         # Time step (ms)
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

    for k in kwargs:
        pars[k] = kwargs[k]

    return pars
# -------------------- Making rate vectors ------------------------------------
def generate_rate_profile(pars):
    t = pars['range_t']
    rates = []

    # Case 1: 10 Hz constant
    rates.append(10)

    # Case 2: 50 Hz constant
    rates.append(50)

    # Case 3: 100 Hz constant
    rates.append(100)

    # Case 4: 100 Hz only from 500–600 ms
    rate_4 = np.zeros_like(t)
    rate_4[(t >= 500) & (t < 600)] = 100
    rates.append(rate_4)

    return rates

# ---------------------- Defining probability functions ----------------------

def release_probability(rate, pars):
    
    # -------------------- Unpacking parameters -------------------------
    Po_dep, Pf_dep, tau_dep = pars['Po_dep'], pars['Pf_dep'], pars['tau_dep']
    Po_fac, Pf_fac, tau_fac = pars['Po_fac'], pars['Pf_fac'], pars['tau_fac']
    dt, t = pars['dt'], pars['range_t']     # dt in milliseconds
    Lt = len(t)
    
    # -------------------- Creating rate vector -------------------------
    # If rate is scalar (e.g. 10 Hz), convert to constant array
    if np.isscalar(rate):
        rate_vec = np.full(Lt, rate)
    else:
        rate_vec = rate  # already a time-dependent array

    # -------------------- Generating spikes ----------------------------
    # Using vectorized method to generate spike train
    spike_train = (np.random.rand(Lt) < rate_vec * dt * 1e-3).astype(int)
    spike_total = np.sum(spike_train)
    print(f"Presynaptic neuron spikes {spike_total} times for rate {rate}")

    # --------- Transmission and Probability for depressive synapse -----
    P_dep = np.zeros(Lt)                 # probability of transmission
    trans_dep = np.zeros(Lt)             # depression synapse transmission  
    P_dep[0] = Po_dep                    # Setting initial value
    
    for i in range(Lt):
        # Predicting transmission given probability
        if spike_train[i] == 1:
            trans_dep[i] = np.random.rand() < P_dep[i]  # Bernoulli draw
            if trans_dep[i] == 1:
                P_dep[i] = 0  # Reset probability after transmission

        # Updating depressive probability
        if i < Lt - 1:
            P_dep[i+1] = P_dep[i] + dt / tau_dep * (Po_dep - P_dep[i])

    # Ensuring P stays within bounds
    P_dep = np.clip(P_dep, Pf_dep, Po_dep)
        
    # --------- Transmission and Probability for facilitating synapse -----
    P_fac = np.zeros(Lt)                # probability of transmission for facilitation 
    trans_fac = np.zeros(Lt)            # facilitation synapse transmission
    P_fac[0] = Po_fac                   # facilitation initial P 
    
    for i in range(Lt):
        # Predicting transmission and increasing P if spike occurs
        if spike_train[i] == 1:
            trans_fac[i] = np.random.rand() < P_fac[i]  # Bernoulli draw
            P_fac[i] = P_fac[i] + 0.1 * (1 - P_fac[i])   # Facilitation rule

        # Updating facilitative probability
        if i < Lt - 1:
            P_fac[i+1] = P_fac[i] + dt / tau_fac * (Po_fac - P_fac[i])
    
    # Ensuring P stays within bounds
    P_fac = np.clip(P_fac, Po_fac, Pf_fac)
        
    return spike_train, P_dep, trans_dep, P_fac, trans_fac



# ----------------------Plotting function -------------------------------------

def plot_results(pars, rate, spike_train, P_dep, trans_dep, P_fac, trans_fac):
    t = pars['range_t']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- Plot 1: Raster Plot ---
    axs[0].plot(t[spike_train == 1], [2]*int(np.sum(spike_train)), 'k|', label='Spike', markersize=8)
    axs[0].plot(t[trans_dep == 1], [1]*int(np.sum(trans_dep)), 'r|', label='Depressing Transmission', markersize=8)
    axs[0].plot(t[trans_fac == 1], [0]*int(np.sum(trans_fac)), 'b|', label='Facilitating Transmission', markersize=8)


    axs[0].set_yticks([0, 1, 2])
    axs[0].set_yticklabels(['Facilitation Tx', 'Depression Tx', 'Spike'])
    axs[0].set_ylabel("Events")
    axs[0].legend(loc="upper right")
    axs[0].set_title(f"Spike Train and Transmission Events (Rate = {rate} Hz)")

    # --- Plot 2: Depressing Synapse Probability ---
    axs[1].plot(t, P_dep, 'r')
    axs[1].set_ylim([0, 1])
    axs[1].set_ylabel("P (Depressing)")
    axs[1].set_title("Release Probability of Depressing Synapse")

    # --- Plot 3: Facilitating Synapse Probability ---
    axs[2].plot(t, P_fac, 'b')
    axs[2].set_ylim([0, 1])
    axs[2].set_ylabel("P (Facilitating)")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_title("Release Probability of Facilitating Synapse")

    plt.tight_layout()
    plt.show()
    
    
# -----------------------Transmission Rate function ---------------------------
def plot_transmission_rate_curve(pars, rate_values, sim_time=5000):
    """
    Simulates different constant firing rates and plots average transmission rates
    for both depressing and facilitating synapses.
    """
    pars['range_t'] = np.arange(0, sim_time, pars['dt'])  # override time
    dt = pars['dt']
    Lt = len(pars['range_t'])

    trans_rate_dep = []
    trans_rate_fac = []

    for r in rate_values:
        spike_train, P_dep, trans_dep, P_fac, trans_fac = release_probability(r, pars)
        dep_rate = np.sum(trans_dep) / (sim_time / 1000)  # transmissions/sec
        fac_rate = np.sum(trans_fac) / (sim_time / 1000)
        trans_rate_dep.append(dep_rate)
        trans_rate_fac.append(fac_rate)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(rate_values, trans_rate_dep, 'r-o', label='Depressing Synapse')
    plt.plot(rate_values, trans_rate_fac, 'b-o', label='Facilitating Synapse')
    plt.xlabel('Presynaptic Firing Rate (Hz)')
    plt.ylabel('Transmission Rate (Hz)')
    plt.title('Synaptic Transmission Rate vs Firing Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ----------------------------Using functions ---------------------------------
rate_profiles = generate_rate_profile(default_pars())

for i, rate in enumerate(rate_profiles):
    print(f"\n--- Condition {i+1}: {'Custom' if isinstance(rate, np.ndarray) else f'{rate} Hz'} ---")
    
    pars = default_pars()
    spike_train, P_dep, trans_dep, P_fac, trans_fac = release_probability(rate, pars)

    plot_results(pars, rate if not isinstance(rate, np.ndarray) else "100Hz (500–600ms)", 
                 spike_train, P_dep, trans_dep, P_fac, trans_fac)

rate_values = np.arange(0, 105, 10)  # from 0 to 100 Hz in steps of 10
plot_transmission_rate_curve(pars, rate_values)
        
        
