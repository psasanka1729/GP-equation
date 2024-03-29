#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import math
import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse
from scipy.sparse import csr_matrix


# # Matplotlib plotting parameters

# In[73]:


# # Real atomtronic potential

# In[74]:


r"""
import os
path = "/Users/sasankadowarah/atomtronics/GP-equation-main"
os.chdir(path)   
position_landscape,potential_landscape= np.loadtxt("potential_landscape_original.txt", delimiter = '\t', unpack=True)
def extract_source_gate_drain_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -34
    # Position where the gate well ends.
    gate_well_end_index   = 60#2.98
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) & (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

transistor_position  = extract_source_gate_drain_well_potential(position_landscape,potential_landscape)[0]
transistor_potential = extract_source_gate_drain_well_potential(position_landscape,potential_landscape)[1]
f = plt.figure()
f.set_figwidth(12)
f.set_figheight(8)
plt.plot(transistor_position,transistor_potential)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
plt.show()""";


# # Gate well potential

# In[75]:


r"""
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

#gate_well_position  = extract_gate_well_potential(position_landscape,potential_landscape)[0]
#gate_well_potential = extract_gate_well_potential(position_landscape,potential_landscape)[1]
gate_well_position_exp  = extract_gate_well_potential(transistor_position,transistor_potential)[0]
gate_well_potential_exp = extract_gate_well_potential(transistor_position,transistor_potential)[1]
f = plt.figure()
plt.plot(gate_well_position_exp,gate_well_potential_exp,linewidth=4.1)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
f.set_figwidth(10)
f.set_figheight(8)
#plt.savefig('source_well.png', dpi=300)
plt.show()""";


# In[76]:


r"""
def reduce_density_points(array):
    new_array = []

  # Iterate over the array, skipping the first and last elements.
    for i in range(1, len(array),2):
    # Calculate the sum of the two nearest neighbours.
        sum_of_neighbours = array[i]

    # Set the corresponding entry in the new array to the sum of the neighbours.
        new_array.append(sum_of_neighbours)

  # Return the new array.
    return new_array

reduced_density_positions = reduce_density_points(gate_well_position_exp)
for i in range(3):
    reduced_density_positions = reduce_density_points(reduced_density_positions)

reduced_density_potentials = reduce_density_points(gate_well_potential_exp)
for i in range(3):
    reduced_density_potentials = reduce_density_points(reduced_density_potentials)


f = plt.figure()
ax = f.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.9)
ax.tick_params(axis="x", direction="inout", length=10, width=1.5, color="k")
ax.tick_params(axis="y", direction="inout", length=10, width=1.5, color="k")
plt.plot(reduced_density_positions, reduced_density_potentials,linewidth=4.1)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
f.set_figwidth(14)
f.set_figheight(8)
plt.show()""";   


# In[77]:


#gate_well_position_exp = np.array(reduced_density_positions)
#gate_well_potential_exp = np.array(reduced_density_potentials)
#len(gate_well_potential_exp)


# # Defines the transistor potential

# In[78]:


def source_well(source_bias_voltage,
                source_start,
                source_end):
    source_position_arr  = np.linspace(-40,source_end,1000)
    source_potential_arr = []
    for x in source_position_arr:
        if  -60 <= x <= source_start:
            # barrier on the left of the source well
            source_potential_arr.append(1000)
        elif source_start <= x <= source_end:
            source_potential_arr.append(source_bias_voltage)
    return source_position_arr,np.array(source_potential_arr)

def gate_well(gate_start,gate_end,gate_bias_voltage):
    gate_position_arr  = np.linspace(gate_start,gate_end,1000)
    gate_potential_arr = []
    for x in gate_position_arr:
        gate_potential_arr.append(gate_bias_voltage)
    return gate_position_arr,gate_potential_arr

def drain_well(drain_start,
              drain_end):
    drain_position_arr  = np.linspace(drain_start,40,1000)
    drain_potential_arr = []
    for x in drain_position_arr:
        if drain_start <= x <= drain_end:
            # the drain is at zero potential
            drain_potential_arr.append(0.0)
        elif drain_end <= x <= 40:
            drain_potential_arr.append(1000)
    return drain_position_arr,np.array(drain_potential_arr)

def step_barrier_between_wells(barrier_start, barrier_end, barrier_height):
    barrier_position_arr  = np.linspace(barrier_start,barrier_end,10)
    barrier_potential_arr = []
    for x in barrier_position_arr:
        barrier_potential_arr.append(barrier_height)
    return barrier_position_arr,barrier_potential_arr

source_well_bias_potential = 20
source_well_start          = -20
source_well_end            = -6

source_gate_barrier_start  = source_well_end
source_gate_barrier_end    = -5

gate_bias_potential        = 0
gate_well_start            = source_gate_barrier_end
gate_well_end              = -2

gate_drain_barrier_start   = gate_well_end
gate_drain_barrier_end     = -1

drain_well_start           = gate_drain_barrier_end
drain_well_end             = 30


SG_barrier_height = 31
GD_barrier_height = 32

# saving potential landscape data to a file for to used later
transistor_landscape_file = open("transistor_landscape.txt","w")
transistor_landscape_file.write(str("source_well_bias")+"\t"+str(source_well_bias_potential)+"\n"+
		str("source_well_start")+"\t"+str(source_well_start)+"\n"+
		str("source_well_end")+"\t"+str(source_well_end)+"\n"+
		str("SG_start")+"\t"+str(source_gate_barrier_start)+"\n"+
		str("SG_end")+"\t"+str(source_gate_barrier_end)+"\n"+
		str("gate_well_bias")+"\t"+str(gate_bias_potential)+"\n"+
		str("gate_well_start")+"\t"+str(gate_well_start)+"\n"+
		str("gate_well_end")+"\t"+str(gate_well_end)+"\n"+
		str("GD_start")+"\t"+str(gate_drain_barrier_start)+"\n"+
		str("GD_end")+"\t"+str(gate_drain_barrier_end)+"\n"+
		str("drain_well_start")+"\t"+str(drain_well_start)+"\n"+
		str("drain_well_end")+"\t"+str(drain_well_end)+"\n"+
		str("SG_barrier_height")+"\t"+str(SG_barrier_height)+"\n"+
		str("GD_barrier_height")+"\t"+str(GD_barrier_height))

transistor_landscape_file.close();



source_well_position,source_well_potential = source_well(source_well_bias_potential,source_well_start,
            source_well_end)
source_gate_barrier_position,source_gate_barrier_potential = step_barrier_between_wells(
                                                             source_gate_barrier_start,
                                                             source_gate_barrier_end,SG_barrier_height)

gate_well_position_exp,gate_well_potential_exp = gate_well(gate_well_start,gate_well_end,
                                                           gate_bias_potential)

gate_drain_barrier_position,gate_drain_barrier_potential = step_barrier_between_wells(
                                                        gate_drain_barrier_start,
                                                        gate_drain_barrier_end,GD_barrier_height)
drain_well_position,drain_well_potential   = drain_well(drain_well_start,drain_well_end)

source_gate_drain_well_position = np.concatenate((source_well_position,
                source_gate_barrier_position,
                gate_well_position_exp,
                gate_drain_barrier_position,
                drain_well_position));

source_gate_drain_well_potential = np.concatenate((source_well_potential,
                source_gate_barrier_potential,
                gate_well_potential_exp,
                gate_drain_barrier_potential,
                drain_well_potential));
r"""
# combining the position and potential arrays.
source_gate_drain_well_position = np.concatenate(
            (source_well_position,
            np.concatenate(
            (gate_well_position_exp,drain_well_position)
            )
            ))
source_gate_drain_well_potential = np.concatenate(
            (source_well_potential,
            np.concatenate(
            (gate_well_potential_exp,drain_well_potential)
            )
            ))""";



# # Source well

# In[79]:


r"""
def extract_source_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -60
    # Position where the gate well ends.
    gate_well_end_index   = B
    # Extracts the gate well position.
    well_position = position[np.where((position >= gate_well_start_index)
                                      & (position <= gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:
                               np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

source_well_position  = extract_source_well_potential(source_gate_drain_well_position,
                                                      source_gate_drain_well_potential)[0]
source_well_potential = extract_source_well_potential(source_gate_drain_well_position,
                                                      source_gate_drain_well_potential)[1]
f = plt.figure()
plt.plot(source_well_position,source_well_potential,linewidth=4.1)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
f.set_figwidth(10)
f.set_figheight(8)
#plt.savefig('source_well.png', dpi=300)
plt.show()""";


# # Transistor parameters for code

# In[80]:


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

index = int(sys.argv[1])
N_atom_lst = np.array([20,30,40,50,60,70,100,200,300,400,500,1000,1200,1500,2000,3000])*10**(3)
N_atom = N_atom_lst[index]

g_source   = (4*PI*H_BAR**2*a_s)/(A*M)


# Length of the space interval.
L  = len(xs)#(max(xs)-min(xs))
# Increment in the space interval.
dx = np.abs(xs[1]-xs[0])
# Increment in momentum space interval.
dk = (2*PI)/L

# Trapping frequency of the harmionic oscillator.
def V_external(x):
    return external_potential


# H(k)
def Hamiltonian_k(p):
    return p**2/(2*M)

# H(x).
def Hamiltonian_x(x,psi):
    return V_external(x)+N_atom*g_source*np.abs(psi)**2

# Momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
E_k = (H_BAR**2*k**2)/(2*M)
# The exact ground state of a quantum harmonic oscillator.
def psi_0(x):
    return (((M*OMEGA/(PI*H_BAR)))**(1/4))*np.exp(-(M*OMEGA/(2*H_BAR))*x**2)


# # Imaginary time evolution code

# In[81]:


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
    #psi_initial = np.sin(xs)
    # Normalize the initial guess wavefunction.
    psi_initial = normalize_x(psi_initial) 
    
    psi_k    = fftpack.fft(psi_initial)
    #psi_x = psi_initial
    
    for i in range(total_iterations):
        
        
        #r"""
        psi_k = np.exp(-(E_k * dt)/(2*H_BAR)) * psi_k
        psi_x = fftpack.ifft(psi_k)
        psi_x = np.exp(-Hamiltonian_x(xs,psi_x) * dt/H_BAR) * psi_x
        psi_k = fftpack.fft(psi_x)
        psi_k = np.exp(-(E_k * dt)/(2*H_BAR)) * psi_k
        psi_x = fftpack.ifft(psi_k)
        psi_x = normalize_x(psi_x)
        psi_k = fftpack.fft(psi_x)#"""
    
    
    psi_k = fftpack.fft(psi_x)
    psi_k = normalize_k(psi_k)
        
    return psi_x
psi_ITE = imaginary_time_evolution(10**(-6),10000)


# # Plot

# In[82]:


r"""
#time_step = 
psi_ITE = imaginary_time_evolution(10**(-6),10000)
# Create some mock data
t = xs/(1.e-6)
data1 = np.abs(psi_ITE)**2*dx
data2 = source_well_potential

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel(r"Position, $\mu $ m")
ax1.set_ylabel(r"$\psi_{0}$", color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axvline(-3.7, color='green',label = "SG boundary")
#plt.legend()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"
ax2.set_ylabel(r"Potential, kHz ", color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color,linewidth = 3)
ax2.tick_params(axis="y", labelcolor=color)
fig.set_figwidth(15)
fig.set_figheight(7)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
#import os
#path = "/Users/sasankadowarah/atomtronics/plots"
#os.chdir(path)
#plt.savefig("ITE_ground_state_2_9_500000.jpg", dpi=600)
plt.show()""";


# In[83]:


source_well_length = len(source_well_potential)


# In[84]:


r"""
chemical_potential = source_well_potential*10**3*2*PI*H_BAR+g_source*N_atom*np.abs(psi_ITE)**2
plt.plot(source_well_position,chemical_potential/(10**3*2*PI*H_BAR))
plt.plot(source_well_position,source_well_potential,color="red",linewidth = 6.1)
plt.show()""";


# # Gate well

# In[85]:


r"""
def extract_gate_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = B
    # Position where the gate well ends.
    gate_well_end_index   = C
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) &
                                      (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:
                               np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

#gate_well_position  = extract_gate_well_potential(position_landscape,potential_landscape)[0]
#gate_well_potential = extract_gate_well_potential(position_landscape,potential_landscape)[1]
gate_well_position  = extract_gate_well_potential(source_gate_drain_well_position,
                                                  source_gate_drain_well_potential)[0]
gate_well_potential = extract_gate_well_potential(source_gate_drain_well_position,
                                                  source_gate_drain_well_potential)[1]
gate_well_length = len(gate_well_potential)
f = plt.figure()
plt.plot(gate_well_position,gate_well_potential,linewidth=4.1)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
f.set_figwidth(10)
f.set_figheight(8)
#plt.savefig('source_well.png', dpi=300)
plt.show()""";


# # Drain well

# In[86]:


r"""
def extract_drain_well_potential(position,potential):
    
    # Position where the gate well starts.
    drain_well_start_index = D
    # Position where the gate well ends.
    drain_well_end_index   = 60
    # Extracts the gate well position.
    well_position = position[np.where((position > drain_well_start_index) 
                                      & (position < drain_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:
                               np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

#drain_well_position  = extract_drain_well_potential(position_landscape,potential_landscape)[0]
#drain_well_potential = extract_drain_well_potential(position_landscape,potential_landscape)[1]
drain_well_position  = extract_drain_well_potential(source_gate_drain_well_position,
                                                    source_gate_drain_well_potential)[0]
drain_well_potential = extract_drain_well_potential(source_gate_drain_well_position,
                                                    source_gate_drain_well_potential)[1]
f = plt.figure()
plt.plot(drain_well_position,drain_well_potential,linewidth=4.1)
plt.xlabel(r"$\mu$ m")
plt.ylabel(r"kHz")
f.set_figwidth(10)
f.set_figheight(8)
#plt.savefig('source_well.png', dpi=300)
plt.show()""";


# # Real time evolution

# In[87]:


"""
The potential is in kHz units. It is converted to SI units
by multiplying 10^3 * h.
"""
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)
external_potential = source_gate_drain_well_potential*(10**3)*2*PI*(H_BAR) # J
xs = source_gate_drain_well_position*1.e-6 # m

N  = len(external_potential)
r""" Rb87 parameters """
M   = 1.4192261*10**(-25) # kg
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614
trap_frequency = 918
trap_length = np.sqrt(H_BAR/(M*trap_frequency)) # m
A = PI*trap_length**2


# Length of the space interval.
L  = len(xs)#(max(xs)-min(xs))
# Increment in the space interval.
dx = np.abs(xs[1]-xs[0])
# Increment in momentum space interval.
dk = (2*PI)/L


# Momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
E_k = (H_BAR**2*k**2)/(2*M)
# The exact ground state of a quantum harmonic oscillator.
def psi_0(x):
    return (((M*OMEGA/(PI*H_BAR)))**(1/4))*np.exp(-(M*OMEGA/(2*H_BAR))*x**2)


# trapping frequency in the source well.
source_trap_frequency = 315#918#315.3884
trap_length = np.sqrt(H_BAR/(M*source_trap_frequency)) # m
A = PI*trap_length**2
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)

# trapping frequency in the gate well.
gate_trap_frequency = 5653#5556.7287
trap_length = np.sqrt(H_BAR/(M*gate_trap_frequency)) # m
A = PI*trap_length**2
g_gate   = (4*PI*H_BAR**2*a_s)/(A*M)

# trapping frequency in the drain well.
drain_trap_frequency = 1113#330.5382
trap_length = np.sqrt(H_BAR/(M*drain_trap_frequency)) # m
A = PI*trap_length**2
g_drain   = (4*PI*H_BAR**2*a_s)/(A*M)

# the interaction g is different in each well
g = np.zeros(N)
for i in range(N):
    if -60<=xs[i]/(1.e-6)<=source_well_end:
        g[i] = g_source
    elif source_well_start<=xs[i]/(1.e-6)<=drain_well_start:
        g[i] = g_gate
    elif xs[i]/(1.e-6)>=drain_well_start:
        g[i] = g_drain

# Trapping frequency of the harmionic oscillator.
def V_external(x):
    return external_potential


# H(k)
def Hamiltonian_k(p):
    return p**2/(2*M)

# H(x).
def Hamiltonian_x(x,psi):
    #g = 0
    return V_external(x)+N_atom*g*np.abs(psi)**2

def normalize_x(wavefunction_x):
    return wavefunction_x/(np.sqrt(np.sum(np.abs(wavefunction_x)**2)*dx))

while len(psi_ITE) < N:
    psi_ITE=np.hstack((psi_ITE,np.array([0])))


# # Runge-Kutta algorithm

# In[88]:


# Laplace Operator (Finite Difference)
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(xs.size, xs.size)) / dx**2
H = - (H_BAR**2/(2*M)) * D2


# In[89]:


if external_potential is not None:
    H += scipy.sparse.spdiags(external_potential,0,N,N)


# In[90]:


gamma_power = 28
gamma = 0#10**(-gamma_power)
atom_removal_term = gamma*np.tanh(source_gate_drain_well_position-10)
#plt.plot(source_gate_drain_well_position,atom_removal_term)
#plt.show()


# In[91]:


def dpsi_dt(t,psi):
    dpsi_dt = (-1j/H_BAR) * (H.dot(psi) + (g)*N_atom*(np.abs(psi)**2)*(psi) - atom_removal_term*psi)
    return dpsi_dt


# In[92]:





# ## Runge kutta with snapshots of atom density at a few fixed times

# In[ ]:


t0 = 0.0
dt = 10**(-8)
number_of_snapshots = 10
def wavefunction_t(total_time):
    import os
    psi_0 = np.complex64(psi_ITE)
    psi_0 = normalize_x(psi_0)
    psi_t = psi_0
    t = t0
    number_of_iterations = int(total_time/dt)
    print("number of iterations = ", number_of_iterations)
    for i in range(number_of_iterations):    
        k1 = dt * dpsi_dt(t, psi_t)
        k2 = dt * dpsi_dt(t + dt/2, psi_t + k1/2)
        k3 = dt * dpsi_dt(t + dt/2, psi_t + k2/2)
        k4 = dt * dpsi_dt(t + dt, psi_t + k3)

        psi_t = psi_t + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
     
    
        snapshot_time_lst = np.linspace(t0,total_time,number_of_snapshots)        
        for snapshots in snapshot_time_lst:
            if np.abs(t - snapshots) < dt:
                np.save("wavefunction_time_"+str(N_atom)+"_"+str(np.around(snapshots*10**(3),2))+".npy",psi_t)

    return psi_t
time_t = 100*10**(-3) # s
time_evolved_wavefunction = wavefunction_t(time_t)


# In[ ]:


snapshot_time_lst = np.around(np.linspace(t0,time_t,number_of_snapshots)*10**(3),2)
np.save("snapshot_time_lst.npy",snapshot_time_lst)


# # Density of atoms as a function of time

# In[ ]:

# In[25]:
  


# ## Saves the heatmap of the wavefunction as a funcion of time

# In[ ]:


# # Heatmap of atom density

# In[ ]:


# In[49]:


# In[94]:



