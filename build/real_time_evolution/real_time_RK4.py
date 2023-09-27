#!/usr/bin/env python
# coding: utf-8

# In[418]:


import math
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse
from scipy.sparse import csr_matrix


# # Imaginary time evolution ground state

# In[419]:


ground_state_position,initial_ground_state_wavefunction= np.loadtxt("ground_state_wavefunction_source_well.txt", delimiter = '\t', unpack=True)

# # Transistor potential

# In[420]:


position_landscape,potential_landscape= np.loadtxt("potential_landscape.txt", delimiter = '\t', unpack=True)
def extract_source_and_gate_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -34
    # Position where the gate well ends.
    gate_well_end_index   = 60#2.98
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) & (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

source_and_gate_well_position  = extract_source_and_gate_well_potential(position_landscape,potential_landscape)[0]
source_and_gate_well_potential = extract_source_and_gate_well_potential(position_landscape,potential_landscape)[1]

# # Reducing density of points

# In[421]:


def replace_with_sum_of_nearest_neighbours(array):
    new_array = []

  # Iterate over the array, skipping the first and last elements.
    for i in range(1, len(array),2):
    # Calculate the sum of the two nearest neighbours.
        sum_of_neighbours = array[i]

    # Set the corresponding entry in the new array to the sum of the neighbours.
        new_array.append(sum_of_neighbours)

  # Return the new array.
    return new_array

new_positions = replace_with_sum_of_nearest_neighbours(source_and_gate_well_position)
for i in range(3):
    new_positions = replace_with_sum_of_nearest_neighbours(new_positions)

new_potentials = replace_with_sum_of_nearest_neighbours(source_and_gate_well_potential)
for i in range(3):
    new_potentials = replace_with_sum_of_nearest_neighbours(new_potentials)

# In[422]:


source_gate_drain_well_position = np.array(new_positions)
source_gate_drain_well_potential = np.array(new_potentials)


# # Source well potential

# In[423]:


def extract_source_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -34
    # Position where the gate well ends.
    gate_well_end_index   = -3.7
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) & (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

source_well_position  = extract_source_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[0]
source_well_potential = extract_source_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[1]
# # Gate well potential

# In[424]:


def extract_gate_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -3.7
    # Position where the gate well ends.
    gate_well_end_index   = 2.98
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) & (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

gate_well_position  = extract_gate_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[0]
gate_well_potential = extract_gate_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[1]
# # Drain well potential

# In[425]:


def extract_drain_well_potential(position,potential):
    
    # Position where the gate well starts.
    drain_well_start_index = 2.98
    # Position where the gate well ends.
    drain_well_end_index   = 60
    # Extracts the gate well position.
    well_position = position[np.where((position > drain_well_start_index) & (position < drain_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

#drain_well_position  = extract_drain_well_potential(position_landscape,potential_landscape)[0]
#drain_well_potential = extract_drain_well_potential(position_landscape,potential_landscape)[1]
drain_well_position  = extract_drain_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[0]
drain_well_potential = extract_drain_well_potential(source_gate_drain_well_position,source_gate_drain_well_potential)[1]
# # Transistor code parameters

# In[426]:


"""
The potential is in kHz units. It is converted to SI units
by multiplying 10^3 * h.
"""
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)
r""" Rb87 parameters """
# Mass of Rb87 atom.
M   = 1.4192261*10**(-25) # kg
# s wave scattering length of Rb87 atom.
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614

# trapping frequency in the source well.
source_trap_frequency = 918
trap_length = np.sqrt(H_BAR/(M*source_trap_frequency)) # m
# Number of atoms in the source well.
N_atom = 20000
A = PI*trap_length**2
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)

# trapping frequency in the gate well.
gate_trap_frequency = 5653
trap_length = np.sqrt(H_BAR/(M*gate_trap_frequency)) # m
A = PI*trap_length**2
g_gate   = (4*PI*H_BAR**2*a_s)/(A*M)

# trapping frequency in the drain well.
drain_trap_frequency = 1113
trap_length = np.sqrt(H_BAR/(M*drain_trap_frequency)) # m
A = PI*trap_length**2
g_drain   = (4*PI*H_BAR**2*a_s)/(A*M)

external_potential = source_gate_drain_well_potential*(10**3)*2*PI*(H_BAR) # J
xs = source_gate_drain_well_position*10**(-6) # m

# Length of the space interval.
L  = len(xs)
# Increment in the space interval.
dx = np.abs(xs[1]-xs[0])
# Increment in momentum space interval.
dk = (2*PI)/L


N  = len(external_potential)

source_gate_drain_well_length = len(new_positions)
source_well_length = len(ground_state_position)
gate_well_length = len(gate_well_position)
drain_well_length = len(drain_well_position)
g_source_well = g_source*np.ones(source_well_length)
g_gate_well = g_gate*np.ones(source_gate_drain_well_length-source_well_length-drain_well_length)
g_drain_well = g_drain*np.ones(source_gate_drain_well_length-source_well_length-gate_well_length)
g_source_gate_drain_well = np.hstack([g_source_well,g_gate_well,g_drain_well])

def normalize_x(wavefunction_x):
    return wavefunction_x/(np.sqrt(np.sum(np.abs(wavefunction_x)**2)*dx))

while len(initial_ground_state_wavefunction) < N:
    initial_ground_state_wavefunction=np.hstack((initial_ground_state_wavefunction,np.array([0])))


# # Time evolution code

# In[427]:


# Laplace Operator (Finite Difference)
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(xs.size, xs.size)) / dx**2
H = - (H_BAR/(2*M)) * D2
if external_potential is not None:
    H += scipy.sparse.spdiags(external_potential,0,N,N)


# In[428]:


def dpsi_dt(t,psi):
    dpsi_dt = -1j * (H.dot(psi) + (g_source_gate_drain_well /H_BAR)*N_atom*(np.abs(psi)**2)*(psi))
    return dpsi_dt


# In[446]:


t0 = 0.0
dt = 10**(-7)
def wavefunction_t(total_time):
    psi_0 = np.complex64(initial_ground_state_wavefunction)
    psi_0 = normalize_x(psi_0)
    psi_t = psi_0
    t = t0
    number_of_iterations = int(total_time/dt)
    for i in range(number_of_iterations):    
        k1 = dt * dpsi_dt(t, psi_t)
        k2 = dt * dpsi_dt(t + dt/2, psi_t + k1/2)
        k3 = dt * dpsi_dt(t + dt/2, psi_t + k2/2)
        k4 = dt * dpsi_dt(t + dt, psi_t + k3)

        psi_t = psi_t + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return psi_t


# In[451]:


#time_evolved_wavefunction = wavefunction_t(100*10**(-3))


# In[435]:


def number_of_atom(wavefunction,a,b):
    return N_atom*np.sum(np.abs(wavefunction[a:b])**2)*dx


# In[452]:



# In[185]:


#gate_atom_number = []
#source_atom_number = []
#drain_atom_number = []

import sys
time_t = float(sys.argv[1])*10**(-3)

wavefunction_time_t = wavefunction_t(time_t)
np.save("wavefunction_"+str(time_t)+".npy",wavefunction_t)

gate_atom_number = number_of_atom(wavefunction_time_t,source_well_length,source_well_length+gate_well_length)
source_atom_number = number_of_atom(wavefunction_time_t,0,source_well_length)
drain_atom_number = number_of_atom(wavefunction_time_t,source_well_length+gate_well_length
                                            ,source_well_length+gate_well_length+drain_well_length)

f = open("number_of_atoms.txt","w")
f.write(str(time_t)+"\t"+str(source_atom_number)+"\t"+str(gate_atom_number)+"\t"+str(drain_atom_number)+"\n")

f.close()
# In[ ]:



