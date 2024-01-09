#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse


# In[25]:


# matplotlib paramet


# ## Source well parameters

# In[26]:


r"""

In this section we will calculate the ground state in the source well through imaginary
time evolution. Below we will create the potential for the source well. It is not 
mecessary to do this seperately; one can create the full transistor potential and then
extract the source well. But here we will start with the source well first. While doing
it this way, one has to be careful about using the same source well positions and parameters
later in the complete transistor potential.


"""


r"""

    Throughout the code we will use SI units.

"""

PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)

# number of discretized intervals in the position array
N = 2**12 # choose a power of two to make things simpler for fast Fourier transform

# barrier height outside the trap.
infinite_barrier_height = 10**20*10**3*2*PI*H_BAR


position_start = -2*1.e-6 # m
position_end   = 5*1.e-6 # m

dx = (position_end-position_start)/N

source_well_bias_potential = 0*10**3*2*PI*H_BAR

source_well_start          = 0*1.e-6 # m
source_well_end            = 3*1.e-6 # m

position_arr  = np.linspace(position_start,position_end,N)

r"""

We will start with a potential array with zeroes then gradually fill it up with
position and potential values mentioned above.

"""
potential_arr = np.zeros(N)

# calculates the number of points between the start of the position to the start of the source well
# and fills the potential array with corresponding potential values.
N_barrier_to_source_start = int(abs(source_well_start - position_start)/dx)
# sets the infinite potential in the source well.
for i in range(N_barrier_to_source_start):
    potential_arr[i] = infinite_barrier_height
    
N_source_start_to_source_end = N_barrier_to_source_start+int(abs(source_well_end - source_well_start)/dx)
# sets the source well bias potential.
for i in range(N_barrier_to_source_start,N_source_start_to_source_end):
    potential_arr[i] = source_well_bias_potential   
    
N_source_well_end_to_position_end = N_source_start_to_source_end + int(abs(position_end-source_well_end)/dx)
for i in range(N_source_start_to_source_end,N_source_well_end_to_position_end):
    potential_arr[i] = infinite_barrier_height    
f = plt.figure()    
plt.plot(position_arr*10**6,potential_arr,linewidth=3)
plt.xlabel(r"$ x (\mu m) $")
plt.ylabel(r"J")
f.set_figwidth(8)
f.set_figheight(6)
plt.show()


# ## Paramerters of the atom and the potential

# In[4]:


r""" Rb87 parameters """
M   = 1.4192261*10**(-25) # kg
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614
trap_frequency = 918 # Hz
trap_length = np.sqrt(H_BAR/(M*trap_frequency)) # m
A = PI*trap_length**2 # m*m

N_atom = 20000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)


# ## time split code

# In[5]:


r"""

This section does the time evolution of a wavefunction with a given extrenal potential
using time split (suzuki-trotter decomposition)

"""

# discretizing the momentum space
L  = (position_end-position_start)
dk = (2*PI)/L

# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
def Hamiltonian_x(position_array,potential_array,psi): # H(x)
    return potential_array+N_atom*g_source*np.abs(psi)**2

# momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
E_k = (H_BAR**2*k**2)/(2*M)

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

    
def time_split_suzukui_trotter(initial_wavefunction,position_array,potential_array,dt,total_time):    
    
    r"""
    
    Input : 1. initial wavefunction
            2. position array of the potential
            3. external potential
            4. time interval dt (in seconds)
            5. final time (assuming t_0 = 0)
            
            
    Output: time evolved wavefunction from t_0 = 0 to t_final = total_time
    
    """
    
    psi_k = fftpack.fft(initial_wavefunction)
    psi_x = initial_wavefunction
    
    total_iterations = int(np.abs(total_time)/np.abs(dt))
    print("Number of iterations =", total_iterations)
    
    for _ in range(total_iterations):

        r"""# VTV
            psi_x = np.exp(-1j*Hamiltonian_x(position_array,potential_array,psi_x) * dt/(2*H_BAR))*psi_x
            psi_x = normalize_x(psi_x)
            psi_k = fftpack.fft(psi_x)
            psi_k = np.exp(-1j*(E_k * dt)/(H_BAR))*psi_k
            psi_x = fftpack.ifft(psi_k)
            psi_x = normalize_x(psi_x)
            psi_x = np.exp(-1j*Hamiltonian_x(position_array,potential_array,psi_x) * dt/(2*H_BAR))*psi_x"""
        
        
        # TVT
        # evolution in k space
        psi_k = np.exp(-(E_k * 1j*dt)/(2*H_BAR)) * psi_k
        
        psi_x = fftpack.ifft(psi_k)
        # evolution in x space
        psi_x = np.exp(-Hamiltonian_x(position_array,potential_array,psi_x) * 1j*dt/H_BAR) * psi_x
        
        psi_x = normalize_x(psi_x)  # normalizing          
        psi_k = fftpack.fft(psi_x)
        # evolution in k space
        psi_k = np.exp(-(E_k * 1j*dt)/(2*H_BAR)) * psi_k
        
        psi_x = fftpack.ifft(psi_k)
        psi_x = normalize_x(psi_x) # normalizing
        
        psi_k = fftpack.fft(psi_x)
    

    psi_x = normalize_x(psi_x) # returns the normalized wavefunction
    
    return psi_x


# ## ground state wavefunction  in the source well

# In[6]:


# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

# wavefunction is evolved in imaginary time to get the ground state
r"""

To evolve the initial wavefunction in imaginary time we replace dt by -i*dt (t_imag = i*t_real).

"""
psi_ITE = time_split_suzukui_trotter(psi_initial,position_arr,potential_arr,-1j*10**(-7),0.1);


# In[27]:




# ## real time evolution of the source well wavefunction

# In[8]:


r"""

In this section we will time evolve the ground state obtained above to verify that
it is really the ground state. For this to be true, the probability density of both
of the wavefunction must be same. This is not a sufficient condition; but it is necessary.
This step can be skipped after it is verified once.

""";

#psi_real_time_source_well = time_split_suzukui_trotter(psi_ITE,position_arr,potential_arr
#                                         ,10**(-7),0.1)


# In[9]:


r""" 

Plots the probability density initial ground state and the time evolved wavefunction.
This step can be skipped after it is verified once.

"""
"""
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(6)
plt.scatter(position_arr*10**6,np.abs(psi_ITE)**2,label="$\psi_{0}(x)$",linewidth=3,color="blue")
plt.plot(position_arr*10**6,np.abs(psi_real_time_source_well)**2,label="$\psi_{t}(x)$",linewidth=3,color="red")
plt.legend()
plt.ylabel(r"$\psi(x)$")
plt.xlabel(r"$x\; (\mu m)$" )
ax = f.gca()
ax.axvline(0.0, color="green",label = r"Infinite well boundary",linestyle="--")
ax.axvline(source_well_end*10**6, color="green",label = r"Infinite well boundary",linestyle="--")
for spine in ax.spines.values():
    spine.set_linewidth(1.9)
plt.gcf().subplots_adjust(bottom=0.2)
#plt.savefig("wavefunction_comparison_initial_ground_state_time_evolved.jpg", dpi=300)
plt.show()#""";


# ## comparison of coherence length of the obtained ground state

# In[10]:


r"""

In this section we will compare the coherence length of our obtained ground state wavefuntion
with the analytical coherence length that exist in literature. To be specific, we will compare
the soliton solution given in Pethick and Smith Section 6.4 Healing of the condensate wave function.
Note that this is a rough comparison.

"""

# the following is extracting the wavefunction value at infinity
# this is required to calibrate the soliton solution we will be using
psi_at_infinity = max(psi_ITE)

# coherence length of the soliton
Xi = H_BAR/np.sqrt(2*M*N_atom*(np.abs(psi_at_infinity)**2)*g_source)


# In[11]:


# soliton solution given in Pethick and Smith Section 6.4 Healing of the condensate wave function
def psi_soliton(x):
    psi = []
    for i in x:
        if i<=0:
            psi.append(0)
        else:
            psi.append(np.tanh(i/(np.sqrt(2)*Xi)))
    return np.array(psi)


# In[28]:



# In[29]:


r"""

In this section we will plot the approximate chemical potential of the obtained ground state
in the source well. We will consider the Thomas Fermi limit where we ignore the kinetic energy.
This can be skipped once it is verifed that the chemical potential is less than the SG barrier height.

"""


# ## transistor potential for time split

# In[30]:


r"""

RK4 runs into errors when the height of the infinite barrier exceeds ~ 10**4*10**3*2*PI*H_BAR.
But for time split to work; we need a very high infinite potentail. So we will define the
transistor potential twice: one with very large infinite barrier for time spit;
and another with a small infinite barrier for RK4.


"""
# height of the infinite barrier in the source and the drain well.
infinite_barrier_height = 10**40*10**3*2*PI*H_BAR

position_start = -2*1.e-6 # this must be same as the position start used earlier in the source well
position_end   = 12*1.e-6 # right end of the transistor potential

# dx must be constant throught the code
N = int((position_end-position_start)/dx) # number of points in the discretized position
print(" Number of divisions in position, N = ",N,"\n")

source_well_bias_potential = 0       # must be same as the source well used earlier
source_well_start          = 0*1.e-6 # must be same as the source well used earlier
source_well_end            = 3*1.e-6 # must be same as the source well used earlier

source_gate_barrier_start  = source_well_end
source_gate_barrier_end    = 3.3*1.e-6 # m

gate_bias_potential        = 0
gate_well_start            = source_gate_barrier_end
gate_well_end              = 4.5*1.e-6 # m

gate_drain_barrier_start   = gate_well_end
gate_drain_barrier_end     = 4.8*1.e-6 # m

drain_well_start           = gate_drain_barrier_end
drain_well_end             = 10*1.e-6 # m

SG_barrier_height = 10*10**3*2*PI*H_BAR
GD_barrier_height = 12*10**3*2*PI*H_BAR

# creates the position array as an equally spaced array.
source_gate_drain_well_position = np.linspace(position_start,position_end,N)
# start with a zero initialized position array of length N.
source_gate_drain_well_potential = np.zeros(N)

r""" 

Seeting the potential array according to the user defined parameters. 

"""

# calculates the number of points between the start of the position to the start of the source well.
N_barrier_to_source_start = int(abs(source_well_start - position_start)/dx)
# sets the infinite potential in the source well.
for i in range(N_barrier_to_source_start):
    source_gate_drain_well_potential[i] = infinite_barrier_height
    
N_source_start_to_source_end = N_barrier_to_source_start+int(abs(source_well_end - source_well_start)/dx)
# sets the source well bias potential.
for i in range(N_barrier_to_source_start,N_source_start_to_source_end):
    source_gate_drain_well_potential[i] = source_well_bias_potential   
    
N_source_well_end_to_gate_well_start = N_source_start_to_source_end+int(abs(gate_well_start-source_well_end)/dx)
# sets the step barrier between the source and the gate well.
for i in range(N_source_start_to_source_end,N_source_well_end_to_gate_well_start):
    source_gate_drain_well_potential[i] = SG_barrier_height
    
N_gate_start_to_gate_end = N_source_well_end_to_gate_well_start+int(abs(gate_well_end-gate_well_start)/dx)
# sets the gate well potential.
for i in range(N_source_well_end_to_gate_well_start,N_gate_start_to_gate_end):
    source_gate_drain_well_potential[i] = gate_bias_potential

N_gate_end_to_drain_well_start = N_gate_start_to_gate_end+int(abs(drain_well_start-gate_well_end)/dx)
# sets the barrier between the gate and the drain well.
for i in range(N_gate_start_to_gate_end,N_gate_end_to_drain_well_start):
    source_gate_drain_well_potential[i] = GD_barrier_height

N_drain_start_to_drain_well_end = N_gate_end_to_drain_well_start+int(abs(drain_well_end-drain_well_start)/dx)
# sets the gate well potential values.
for i in range(N_gate_end_to_drain_well_start,N_drain_start_to_drain_well_end):
    source_gate_drain_well_potential[i] = 0 # drain well is always at zero potential.

N_drain_end_to_barrier = N_drain_start_to_drain_well_end+int(abs(position_end-drain_well_end)/dx)
# sets the infinite barrier after the drain well ends.
for i in range(N_drain_start_to_drain_well_end,N_drain_end_to_barrier):
        source_gate_drain_well_potential[i] = infinite_barrier_height
               


# ### time split to evolve the wavefunction in real time

# In[15]:


# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
if len(k) != N: # this is not necessary if N is even
    k = np.hstack([np.arange(0,N/2), np.arange(-N/2+1,0)])*dk

E_k = (H_BAR**2*k**2)/(2*M)

r"""

To use the initial ground state obtained earlier; we have to extend it to the
full transistor potential by setting it as zero outside the source well.

"""
# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))


# ## transistor potential for RK4

# In[31]:


r"""

RK4 runs into errors when the height of the infinite barrier exceeds ~ 10**4*10**3*2*PI*H_BAR.
But for time split to work; we need a very high infinite potentail. So we will define the
transistor potential twice: one with very large infinite barrier for time spit;
and another with a small infinite barrier for RK4.


"""
# height of the infinite barrier in the source and the drain well.
infinite_barrier_height = 10**3*10**3*2*PI*H_BAR

position_start = -2*1.e-6 # this must be same as the position start used earlier in the source well
position_end   = 12*1.e-6 # right end of the transistor potential

# dx must be constant throught the code
N = int((position_end-position_start)/dx) # number of points in the discretized position
print(" Number of divisions in position, N = ",N,"\n")

source_well_bias_potential = 0       # must be same as the source well used earlier
source_well_start          = 0*1.e-6 # must be same as the source well used earlier
source_well_end            = 3*1.e-6 # must be same as the source well used earlier

source_gate_barrier_start  = source_well_end
source_gate_barrier_end    = 3.3*1.e-6 # m

gate_bias_potential        = 0
gate_well_start            = source_gate_barrier_end
gate_well_end              = 4.5*1.e-6 # m

gate_drain_barrier_start   = gate_well_end
gate_drain_barrier_end     = 4.8*1.e-6 # m

drain_well_start           = gate_drain_barrier_end
drain_well_end             = 10*1.e-6 # m

SG_barrier_height = 10*10**3*2*PI*H_BAR
GD_barrier_height = 12*10**3*2*PI*H_BAR

# creates the position array as an equally spaced array.
source_gate_drain_well_position = np.linspace(position_start,position_end,N)
# start with a zero initialized position array of length N.
source_gate_drain_well_potential = np.zeros(N)

r""" 

Seeting the potential array according to the user defined parameters. 

"""

# calculates the number of points between the start of the position to the start of the source well.
N_barrier_to_source_start = int(abs(source_well_start - position_start)/dx)
# sets the infinite potential in the source well.
for i in range(N_barrier_to_source_start):
    source_gate_drain_well_potential[i] = infinite_barrier_height
    
N_source_start_to_source_end = N_barrier_to_source_start+int(abs(source_well_end - source_well_start)/dx)
# sets the source well bias potential.
for i in range(N_barrier_to_source_start,N_source_start_to_source_end):
    source_gate_drain_well_potential[i] = source_well_bias_potential   
    
N_source_well_end_to_gate_well_start = N_source_start_to_source_end+int(abs(gate_well_start-source_well_end)/dx)
# sets the step barrier between the source and the gate well.
for i in range(N_source_start_to_source_end,N_source_well_end_to_gate_well_start):
    source_gate_drain_well_potential[i] = SG_barrier_height
    
N_gate_start_to_gate_end = N_source_well_end_to_gate_well_start+int(abs(gate_well_end-gate_well_start)/dx)
# sets the gate well potential.
for i in range(N_source_well_end_to_gate_well_start,N_gate_start_to_gate_end):
    source_gate_drain_well_potential[i] = gate_bias_potential

N_gate_end_to_drain_well_start = N_gate_start_to_gate_end+int(abs(drain_well_start-gate_well_end)/dx)
# sets the barrier between the gate and the drain well.
for i in range(N_gate_start_to_gate_end,N_gate_end_to_drain_well_start):
    source_gate_drain_well_potential[i] = GD_barrier_height

N_drain_start_to_drain_well_end = N_gate_end_to_drain_well_start+int(abs(drain_well_end-drain_well_start)/dx)
# sets the gate well potential values.
for i in range(N_gate_end_to_drain_well_start,N_drain_start_to_drain_well_end):
    source_gate_drain_well_potential[i] = 0 # drain well is always at zero potential.

N_drain_end_to_barrier = N_drain_start_to_drain_well_end+int(abs(position_end-drain_well_end)/dx)
# sets the infinite barrier after the drain well ends.
for i in range(N_drain_start_to_drain_well_end,N_drain_end_to_barrier):
        source_gate_drain_well_potential[i] = infinite_barrier_height
               


# ### RK4 algorithm for time evolution in real time

# In[17]:


# laplace Operator (Finite Difference)
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(source_gate_drain_well_position.size, source_gate_drain_well_position.size)) / dx**2

# kinetic part of the Hamiltonian
Hamiltonian = - (H_BAR**2/(2*M)) * D2

# external potential added to the Hamiltonian
if source_gate_drain_well_potential is not None:
    Hamiltonian += scipy.sparse.spdiags(source_gate_drain_well_potential,0,N,N)

r"""

The Gross Pitaevskii equation is written as d\psi /dt = f(\psi)

"""
# we are using the same interaction strength g_source as earlier
# this can be changed easily
# this assumes that all wells have the same trapping frequency
def dpsi_dt(t,psi):
    dpsi_dt = (-1j/H_BAR) * (Hamiltonian.dot(psi) + g_source*N_atom*(np.abs(psi)**2)*psi)
    return dpsi_dt

t0 = 0.0 # initial time

delta_t_lst = [10**(-8),10**(-9),10**(-10)]
import sys
delta_t_index = int(sys.argv[1])
dt = delta_t_lst[delta_t_index] # time step used in Runge Kutta of order four

def wavefunction_t(total_time):
    # initial wavefunction
    psi_0 = np.complex64(psi_initial_for_full_potential)
    psi_0 = normalize_x(psi_0)
    
    psi_t = psi_0
    t = t0
    
    number_of_iterations = int(total_time/dt)
    print("Number of iterations = ",number_of_iterations)
    
    for _ in range(number_of_iterations):   
        
        k1 = dt * dpsi_dt(t, psi_t)
        k2 = dt * dpsi_dt(t + dt/2, psi_t + k1/2)
        k3 = dt * dpsi_dt(t + dt/2, psi_t + k2/2)
        k4 = dt * dpsi_dt(t + dt, psi_t + k3)

        psi_t = psi_t + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
        
    return psi_t

# final time
time_t = 5 # ms
time_evolved_wavefunction_rk4 = wavefunction_t(time_t*10**(-3))

np.save("rk4_wavefunction.npy",time_evolved_wavefunction_rk4)