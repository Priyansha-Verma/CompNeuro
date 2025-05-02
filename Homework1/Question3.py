#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:03:14 2025

@author: pverma
"""
import math 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks 

# --------------defining gating functions
def alpha_n(V):
    num = 0.01 * (V + 55)
    den = 1 - math.exp(-0.1 * (V + 55))
    alpha_n = num/den
    return alpha_n

def beta_n(V):
    beta_n = 0.125 * math.exp(-0.0125 * (V + 65))
    
    return beta_n

def alpha_m(V):
    num = 0.1 * (V + 40)
    den = 1 - math.exp(-0.1 * (V + 40))
    alpha_m = num/den
    return alpha_m

def beta_m(V):
    beta_m = 4.0 * math.exp(-0.0556 * (V + 65))
    
    return beta_m

def alpha_h(V):
    alpha_h = 0.07 * math.exp(-0.05 * (V + 65))
    
    return  alpha_h 

def beta_h(V):
    beta_h = 1/(1 + math.exp(-0.1 * (V + 35))) 
    
    return beta_h 

# ------------------------------------------

def dn(V, n, dt=0.01):
    dn = ((alpha_n(V) * (1-n)) - (n * beta_n(V)))*dt
    
    return dn


def dm(V, m, dt=0.01):
    dm = ((alpha_m(V) * (1-m)) - (m * beta_m(V)))*dt
    
    return dm


def dh(V, h, dt=0.01):
    dh = ((alpha_h(V) * (1-h)) - (h * beta_h(V)))*dt
    
    return dh

#---------------------------------------------

def dv(V, g_L=0.003, E_l=-54.387, g_K=0.36, n=0.3177, E_k=-77, g_Na=1.2,
       m=0.0529, h=0.5961, E_Na=50, Ie_A=0.2, c_m=0.01, dt=0.01): 
    
    I_leak = g_L * (V - E_l)
    I_K = g_K * n**4 * (V - E_k)
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_total = I_leak + I_K + I_Na
    return ((-I_total + Ie_A) / c_m) * dt

#----------------------------# Initial conditions


V = -65.0
m = 0.0529
h = 0.5961
n = 0.3177

dt = 0.01
t_max = 50  # ms
steps = int(t_max / dt)

# Storage for plotting
T, Vs, ms, hs, ns = [], [], [], [], []

# Simulation loop
for i in range(steps):
    time = i * dt
    T.append(time)
    Vs.append(V)
    ms.append(m)
    hs.append(h)
    ns.append(n)

    V += dv(V, n=n, m=m, h=h, dt=dt)
    m += dm(V, m, dt=dt)
    h += dh(V, h, dt=dt)
    n += dn(V, n, dt=dt)



# Plotting: Four subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axs[0].plot(T, Vs, color='tab:red')
axs[0].set_ylabel('V (mV)')
axs[0].set_title('Membrane Voltage')

axs[1].plot(T, ms, color='tab:blue')
axs[1].set_ylim(0,1)
axs[1].set_ylabel('m')
axs[1].set_title('Sodium Activation (m)')


axs[2].plot(T, hs, color='tab:green')
axs[2].set_ylim(0,1)
axs[2].set_ylabel('h')
axs[2].set_title('Sodium Inactivation (h)')

axs[3].plot(T, ns, color='tab:purple')
axs[3].set_ylim(0,1)
axs[3].set_ylabel('n')
axs[3].set_xlabel('Time (ms)')
axs[3].set_title('Potassium Activation (n)')

plt.tight_layout()
plt.show()


# ---------------------------- PART 2------------------------------------------

T = 5000 
dt = 0.01
steps = int(T / dt)
time = np.arange(0, T, dt)

Ie_A = np.arange(0, 501, step = 25)*10**(-3) # 
n_currents = len(Ie_A)
Vs = np.zeros((n_currents, steps))

for i, I in enumerate(Ie_A): 
    V = -65.0 # initializing V 
    m = 0.0529
    h = 0.5961
    n = 0.3177

    for step in range(steps): 
        Vs[i, step] = V
        V += dv(V, n=n, m=m, h=h, dt=dt, Ie_A=I) # Ie_A in pA
        m += dm(V, m, dt=dt)
        h += dh(V, h, dt=dt)
        n += dn(V, n, dt=dt)
        
# Finding Peaks
Spikes = [] 
# firing rate
for row in range(n_currents):
    count = 0
    for t in range(1, steps):
        if Vs[row, t-1] < 0 and Vs[row, t] >=0:
            count += 1 
    Spikes.append(count)
    
Firing_rate = [x/0.5 for x in Spikes] 
    
# Plotting 
plt.figure(figsize=(6, 5))
plt.plot(Ie_A, Firing_rate , 'o-', label='Simulated rate')
plt.ylabel('firing rate (Hz')
plt.xlabel('Current pA')
    
        
 # -------------------------- PART 3 ------------------------------------------
V = -65.0
m = 0.0529
h = 0.5961
n = 0.3177
   
T = 25 # ms
steps = int(T / dt) 
time = np.arange(0, T, dt)

I = np.zeros(steps)
I[:500] = -50*10**(-3) 

Vs = []
for i in I:
    Vs.append(V)
    V += dv(V, n=n, m=m, h=h, dt=dt, Ie_A=i)
    m += dm(V, m, dt=dt)
    h += dh(V, h, dt=dt)
    n += dn(V, n, dt=dt)

#Plotting
plt.figure()
plt.plot(time, Vs)
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')
plt.show()


