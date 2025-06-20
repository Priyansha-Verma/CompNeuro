#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:36:26 2025

@author: pverma
"""

import numpy as np
import matplotlib.pyplot as plt

# Common Parameters
N = 50          # number of neurons in network
A = 40          # Hz
eps = 0.1       
theta_cue = 0.
theta_0 = 0.
a = 2
tau = 5       # ms

# Time
T = 1000.
dt = 0.1
time = np.arange(0, T, dt)

# Neuron preferred orientations
theta_i = np.pi / N * np.linspace(1, N, num=N) - np.pi / 2
d_theta = np.pi / N             # constant because evenly distributed 

# Contrast values
contrast_values = [0.1, 0.2, 0.4, 0.8]

# Threshold-linear function
def f(I_theta):
    return np.maximum(0, I_theta)

# ------------- Simulation Function -------------
def run_simulation(J0, J2):
    r_results = []

    for c in contrast_values:
        # External input
        h_theta = A * c * (1 - eps + eps * np.cos(2 * (theta_i - theta_cue)))

        # Initial firing rate bump
        r = np.maximum(0, a * np.cos(2 * (theta_i - theta_0)))

        for t in time:
            # Recurrent input
            recurrent_input = np.zeros(N)
            for i in range(N):
                recurrent_input[i] = np.sum(
                    (J0 + J2 * np.cos(2 * (theta_i[i] - theta_i))) * r
                ) * d_theta / np.pi

            I_theta = h_theta + recurrent_input
            drdt = (-r + f(I_theta)) / tau
            r = r + dt * drdt

        r_results.append(r)

    return r_results

# ------------- Part 1: J0 = -0.5, J2 = 1 -------------
J0_1, J2_1 = -0.5, 1
r_results_1 = run_simulation(J0_1, J2_1)

plt.figure(figsize=(10, 5))
for i, c in enumerate(contrast_values):
    plt.plot(theta_i, r_results_1[i], label=f'c = {c}')
plt.title(f'Steady-State Activity (J0 = {J0_1}, J2 = {J2_1})')
plt.xlabel('Preferred Orientation θ')
plt.ylabel('Firing Rate r(θ)')
plt.legend()
plt.show()

# ------------- Part 2: J0 = -7.3, J2 = 11 -------------
J0_2, J2_2 = -7.3, 11
r_results_2 = run_simulation(J0_2, J2_2)

plt.figure(figsize=(10, 5))
for i, c in enumerate(contrast_values):
    plt.plot(theta_i, r_results_2[i], label=f'c = {c}')
plt.title(f'Steady-State Activity (J0 = {J0_2}, J2 = {J2_2})')
plt.xlabel('Preferred Orientation θ')
plt.ylabel('Firing Rate r(θ)')
plt.legend()
plt.show()
