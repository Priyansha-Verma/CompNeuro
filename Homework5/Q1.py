#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:11:44 2025

@author: pverma
"""

import numpy as np 
import matplotlib.pyplot as plt 

N = 100                         # no. of units in network
n_iter = 5                      # no. of iterations
q0 = np.arange(0, 1.1, 0.1)     # overlap percentage   
P_list = [1, 5, 10]             # no. of memories stored in network simultaneously 
T = 10

# Step 1: Defining memory patterns - P matrix  
def memory_patterns(P_list, N):
    memory_patterns = {}
    for P in P_list:
        U = np.random.choice([-1, 1], size=(P, N))
        memory_patterns[P] = U 
    return memory_patterns   
    
        
    
# Step 2: Constructing Weight Matrix - M
  
def weight_matrix(p):
    M = np.zeros((N, N))
    for i in range(0, N-1):
        for j in range(i+1, N):
            sum_across_p = np.sum([p[a][i]*p[a][j] for a in range(len(p))])
            if sum_across_p >= 0 :
                M[i,j], M[j, i] = 1, 1
            else :
                M[i,j], M[j, i] = -1, -1          
    return M 

# Step 3: Generating s(0) based on q(0) 

def s0_q0(u_a, q0):
#    u_a = p[1]
    s0 = np.zeros((1,N))
    for i in range(0,N):
        if np.random.rand() < q0:
            s0[0, i] = u_a[i]
        else:
            s0[0, i] = np.random.choice([-1, 1])
    return s0
    
# Step 4: Updating s(t) based on weight matrix
def run_network(M, s0, u_a, T):
    s_t = s0.copy()
    q_t = []

    for t in range(T):
        s_new = np.zeros_like(s_t)
        
        for i in range(N):
            net_input = 0 
            for j in range(N):
                net_input += M[i][j] * s_t[0][j]
            s_new[0][i] = 1 if net_input >= 0 else -1

        s_t = s_new.copy()

        # Compute overlap q(t)
        q = 0
        for i in range(N):
            q += s_t[0][i] * u_a[i]
        q = q / N
        q_t.append(q)

    return q_t 


# Step 5: Run simulations and plot q(0) to q(T)
patterns = memory_patterns(P_list, N)
for P in P_list:
    U = patterns[P]
    M = weight_matrix(U)
    u_a = U[0]  # retrieval of the first pattern

    plt.figure(figsize=(8, 5))
    
    for q0_val in q0:
        s0 = s0_q0(u_a, q0_val)
        q0_actual = sum(s0[0][i] * u_a[i] for i in range(N)) / N
        q_t = run_network(M, s0, u_a, T)
        q_t_full = [q0_actual] + q_t

        plt.plot(range(T+1), q_t_full, label=f"q₀ = {q0_val:.1f}")
        
    plt.title(f"Overlap q(t) from t = 0 to {T}, for P = {P}")
    plt.xlabel("Time step t=1")
    plt.ylabel("q(t)")
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.show()

# Step 6: Stimulating the q(t) values for n_trials to check for memory failure
T = 5
P_list = list(range(1, 51))  
n_trials = 100                  # number of simulations
q_map = np.zeros((len(P_list), T))  

for i, P in enumerate(P_list):
    q_trials = []

    for _ in range(n_trials):
        # Generate P memory patterns
        U = np.random.choice([-1, 1], size=(P, N))
        M = weight_matrix(U)
        u_a = U[0]

        # Perfect initial overlap
        s0 = u_a.reshape(1, N)

        # Run and collect q(t)
        q_t = run_network(M, s0, u_a, T)
        q_trials.append(q_t)

    # Average q(t) over trials
    q_map[i, :] = np.mean(q_trials, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(q_map, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
plt.colorbar(label="q")
plt.xlabel("time step")
plt.ylabel("P")
plt.title("Average q(t) heatmap for q₀ = 1.0 (100 trials per P)")
plt.show()


