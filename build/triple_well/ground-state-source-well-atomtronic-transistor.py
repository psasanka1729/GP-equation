#!/usr/bin/env python
# coding: utf-8

# # Ground state in the source well using imaginary time evolution

# In[102]:


import math
import numpy as np
from math import sqrt
from scipy import fftpack


# In[103]:


x,y= np.loadtxt("potential_landscape.txt", delimiter = '\t', unpack=True)



# In[104]:


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

source_well_position  = extract_source_well_potential(x,y)[0]
source_well_potential = extract_source_well_potential(x,y)[1]



# In[105]:




# In[106]:


"""
The potential is in kHz units. It is converted to SI units
by multiplying 10^3 * h.
"""
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)
external_potential = source_well_potential*(10**3)*2*PI*(H_BAR) # J
xs = source_well_position*1.e-6 # m

N  = len(external_potential)
r""" Rb87 parameters """
M   = 1.4192261*10**(-25) # kg
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614
trap_frequency = 918
trap_length = np.sqrt(H_BAR/(M*trap_frequency)) # m
A = PI*trap_length**2
N_atom = 20*10**4
g   = (4*PI*H_BAR**2*a_s)/(A*M)


# Length of the space interval.
L  = len(xs)#(max(xs)-min(xs))
# Increment in the space interval.
dx = np.abs(xs[1]-xs[0])
# Increment in momentum space interval.
dk = (2*PI)/L

# Trapping frequency of the harmionic oscillator.
OMEGA = 5748.55# 4918.11
def V_external(x):
    return external_potential
    #return 0.5*M*(OMEGA**2)*(x)**2

# H(k)
def Hamiltonian_k(p):
    return p**2/(2*M)

# H(x).
def Hamiltonian_x(x,psi):
    #g = 0
    return V_external(x)+N_atom*g*np.abs(psi)**2


# In[107]:



# In[109]:


# Momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
E_k = (H_BAR**2*k**2)/(2*M)


# In[110]:


# The exact ground state of a quantum harmonic oscillator.
def psi_0(x):
    return (((M*OMEGA/(PI*H_BAR)))**(1/4))*np.exp(-(M*OMEGA/(2*H_BAR))*x**2)


# In[111]:


r"""

A custom normalization function is needed to
ensure correct normalization done using a sum instead of an integral.

"""
# Normalize the wavefunction in real space.
def normalize_x(wavefunction_x):
    return wavefunction_x/(np.sqrt(np.sum(np.abs(wavefunction_x)**2)*dx))
# Normalize the wavefunction in momentum space.
def normalize_k(wavefunction_k):
    return wavefunction_k/(np.sqrt(np.sum(np.abs(wavefunction_k)**2)*dk))

    
def imaginary_time_evolution(dt,total_iterations):    
    
    r"""
    
    Input : time interval: dt and total number of iterations to be performed.
    Output: an ordered pair of imaginary time evolved wavefunction and ground
            state energy calculated using the obtained wavefunction.
    
    """
    # Initial guess for the wavefunction.
    psi_initial = np.ones(N)
    # Normalize the initial guess wavefunction.
    psi_initial = normalize_x(psi_initial) 
    
    #psi_k    = fftpack.fft(psi_initial)
    psi_x = psi_initial
    
    for i in range(total_iterations):

        psi_x = np.exp(-Hamiltonian_x(xs,psi_x) * dt/(2*H_BAR)) * psi_x
        psi_x = normalize_x(psi_x)
        psi_k = fftpack.fft(psi_x)
        psi_k = np.exp(-(E_k * dt)/H_BAR) * psi_k
        psi_x = fftpack.ifft(psi_k)
        psi_x = np.exp(-Hamiltonian_x(xs,psi_x) * dt/(2*H_BAR)) * psi_x
        psi_x = normalize_x(psi_x)        
        
        
        
        r"""
        psi_k = np.exp(-(E_k * dt)/(2*H_BAR)) * psi_k
        psi_x = fftpack.ifft(psi_k)
        psi_x = np.exp(-Hamiltonian_x(xs,psi_x) * dt/H_BAR) * psi_x
        psi_k = fftpack.fft(psi_x)
        psi_k = np.exp(-(E_k * dt)/(2*H_BAR)) * psi_k
        psi_x = fftpack.ifft(psi_k)
        psi_x = normalize_x(psi_x)
        psi_k = fftpack.fft(psi_x)"""
    
    # Taking the absolute value to remove the relative phase in the wavefunction.
    #psi_x = np.abs(fftpack.ifft(psi_k))
    psi_x = normalize_x(psi_x)
    potential_energy = np.sum(V_external(xs)*np.abs(psi_x)**2)*dx
    
    psi_k = fftpack.fft(psi_x)
    psi_k = normalize_k(psi_k)
    psi_k = fftpack.fftshift(psi_k)
    
    kinetic_energy   = np.sum(E_k*np.abs(psi_k)**2)*dk
    
    ground_state_energy = kinetic_energy + potential_energy
        
    return psi_x, ground_state_energy


# In[112]:


psi_ITE = imaginary_time_evolution(10**(-6),10**8)


# In[113]:



psi_ITE_data_file = open("ground_state_wavefunction_source_well.txt","w")
for i in range(N):
    psi_ITE_data_file.write(str(xs[i])+"\t"+str(psi_ITE[0][i].real)+"\n")
psi_ITE_data_file.close()


# In[114]:



# In[115]


# # Numerical GP equation

# In[6]:



# In[ ]:




