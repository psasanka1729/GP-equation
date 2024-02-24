# %
import sys
# import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse


# %%
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)

r""" Rb87 parameters """
M   = 1.4192261*10**(-25) # kg
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614
trap_frequency = 918 # Hz
omega_x = trap_frequency
trap_length = np.sqrt(H_BAR/(M*trap_frequency)) # m
A = PI*trap_length**2 # m*m

N_atom = 10000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)
a_0 = np.sqrt(H_BAR/(trap_frequency*M))
#x_s = a_0
x_s = (4*np.pi*a_s*N_atom*a_0**4)**(1/5)    

epsilon = (H_BAR/(M*omega_x*x_s**2))

delta   = (g_source*N_atom*(x_s**2))/(a_0**3*H_BAR*omega_x)


# %%
original_transistor_position, original_transistor_potential= np.loadtxt("alan_potential_landscape.txt", delimiter = '\t', unpack=True)

# %%
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

# making position dimensionless
original_transistor_position  = (original_transistor_position*1.e-6)/x_s
# changing potential to SI units
original_transistor_potential = (original_transistor_potential*10**3*H_BAR)

new_positions = replace_with_sum_of_nearest_neighbours(original_transistor_position)
for i in range(1):
    new_positions = replace_with_sum_of_nearest_neighbours(new_positions)

new_potentials = replace_with_sum_of_nearest_neighbours(original_transistor_potential)
for i in range(1):
    new_potentials = replace_with_sum_of_nearest_neighbours(new_potentials)

original_transistor_position = np.array(new_positions)
original_transistor_potential = np.array(new_potentials)


# %% [markdown]
# #### Source well potential

# %%
def extract_source_well_potential(position,potential):
    
    # Position where the gate well starts.
    source_well_start_index = -60*10**(-6)/x_s
    # Position where the gate well ends.
    source_well_end_index   = -3.65*10**(-6)/x_s
    # Extracts the gate well position.
    well_position = position[np.where((position > source_well_start_index) & (position < source_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

source_well_position  = extract_source_well_potential(original_transistor_position, original_transistor_potential)[0]
source_well_potential = extract_source_well_potential(original_transistor_position, original_transistor_potential)[1]

# %% [markdown]
# #### Gate well potential

# %%
def extract_gate_well_potential(position,potential):
    
    # Position where the gate well starts.
    gate_well_start_index = -3.65*10**(-6)/x_s
    # Position where the gate well ends.
    gate_well_end_index   = 3.3*10**(-6)/x_s
    # Extracts the gate well position.
    well_position = position[np.where((position > gate_well_start_index) & (position < gate_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

gate_well_position  = extract_gate_well_potential(original_transistor_position, original_transistor_potential)[0]
gate_well_potential = extract_gate_well_potential(original_transistor_position, original_transistor_potential)[1]

# %% [markdown]
# #### Drain well potential

# %%
def extract_drain_well_potential(position,potential):
    
    # Position where the gate well starts.
    drain_well_start_index = 3.3*10**(-6)/x_s
    # Position where the gate well ends.
    drain_well_end_index   = 60*10**(-6)/x_s
    # Extracts the gate well position.
    well_position = position[np.where((position > drain_well_start_index) & (position < drain_well_end_index))]
    # Extract the corresponding potential values for the gate well.
    well_potential = potential[np.where(position == well_position[0])[0][0]:np.where(position == well_position[-1])[0][0]+1]
    
    return [well_position,well_potential]

drain_well_position  = extract_drain_well_potential(original_transistor_position, original_transistor_potential)[0]
drain_well_potential = extract_drain_well_potential(original_transistor_position, original_transistor_potential)[1]

# %% [markdown]
# ## time split code

# %%
N = len(source_well_position)

# discretizing the momentum space
L  = np.abs((source_well_position[-1]-source_well_position[0]))

dx = L/N

dk = (2*PI)/L

# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
def Hamiltonian_x(potential_array,psi): # H(x)
    return potential_array/(epsilon*M*omega_x**2*x_s**2)+delta*epsilon**(3/2)*np.abs(psi)**2

# momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
if len(k) != N:
    k = np.hstack([np.arange(0,N/2), np.arange(-N/2+1,0)])*dk
E_k = k**2*epsilon/2

# Normalize the wavefunction in real space.
def normalize_x(wavefunction_x):
    return wavefunction_x/(np.sqrt(np.sum(np.abs(wavefunction_x)**2)*dx))

# Normalize the wavefunction in momentum space.
def normalize_k(wavefunction_k):
    return wavefunction_k/(np.sqrt(np.sum(np.abs(wavefunction_k)**2)*dk))

    
def time_split_suzukui_trotter(initial_wavefunction,potential_array,dt,total_time):    

    psi_k = fftpack.fft(initial_wavefunction)
    psi_x = initial_wavefunction
    
    total_iterations = int(np.abs(total_time)/np.abs(dt))
    print("Number of iterations =", total_iterations)
    
    for _ in range(total_iterations):

        # TVT
        # evolution in k space
        psi_k = np.exp(-(E_k * 1j*dt)/2) * psi_k
        
        psi_x = fftpack.ifft(psi_k)
        # evolution in x space
        psi_x = np.exp(-Hamiltonian_x(potential_array,psi_x) * 1j*dt) * psi_x
        
        psi_x = normalize_x(psi_x)  # normalizing          
        psi_k = fftpack.fft(psi_x)
        # evolution in k space
        psi_k = np.exp(-(E_k * 1j*dt)/2) * psi_k
        
        psi_x = fftpack.ifft(psi_k)
        psi_x = normalize_x(psi_x) # normalizing
        
        psi_k = fftpack.fft(psi_x)
    

    psi_x = normalize_x(psi_x) # returns the normalized wavefunction
    
    return psi_x

# %% [markdown]
# ## ground state wavefunction  in the source well

# %%
# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

time_exponent = int(sys.argv[1])

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 1.e-2
time_step_SI  = -1j*10**(-7)   
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time);

data0 = source_well_position
data1 = np.abs(psi_source_well_ITE)**2
data3 = source_well_potential/(epsilon*M*omega_x**2*x_s**2)

np.save("source_well_position.npy",data0)
np.save("ground_state_source_well_imaginary_time_evolution.npy", data1)
np.save("dimensionless_source_well_potential.npy",data3)

# %% [markdown]
# ## complete transistor potential landscape

# %%
complete_transistor_position = np.concatenate((source_well_position, gate_well_position, drain_well_position))
complete_transistor_potential = np.concatenate((source_well_potential, gate_well_potential, drain_well_potential))

N = len(complete_transistor_position)

# discretizing the momentum space
L  = np.abs((complete_transistor_position[-1]-complete_transistor_position[0]))

dx = L/N

dk = (2*PI)/L

# %% [markdown]
# # time split to evolve the wavefunction in real time

# %%
# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    
time_lst = np.linspace(0,50,16)    
time_index = int(sys.argv[1])

final_time_SI = time_lst[time_index]*10**(-3)
time_step_SI  = 10**(-7)  
# time is made dimensionless  
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential,
                                        time_step,final_time)

data0 = complete_transistor_position
data1 = N_atom*np.abs(time_evolved_wavefunction_time_split)**2*dx
data3 = complete_transistor_potential/(epsilon*M*omega_x**2*x_s**2)

np.save("complete_transistor_position.npy", data0)
np.save("real_time_evolution_time_split.npy", data1)
np.save("dimentionless_complete_transistor_potential.npy", data3)
# %%
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(complete_transistor_position.size, complete_transistor_position.size)) / dx**2

# kinetic part of the Hamiltonian
Hamiltonian = - (epsilon/2) * D2

# dimensionless external potential added to the Hamiltonian
#if source_gate_drain_well_potential is not None:
#    Hamiltonian += scipy.sparse.spdiags(source_gate_drain_well_potential/(epsilon*M*omega_x**2*x_s**2),0,N,N)

def dpsi_dt(t,psi):
    return -1j*(Hamiltonian.dot(psi) + 
                (complete_transistor_potential/(epsilon*M*omega_x**2*x_s**2))*psi + 
                delta*epsilon**(3/2)*np.abs(psi)**2*psi)

t0 = 0.0
#time_step_SI  = 10**(-8-time_exponent)    
#time_step = omega_x*time_step_SI
def wavefunction_t(total_time):
    # initial wavefunction
    psi_0 = np.complex64(psi_initial_for_full_potential)
    psi_0 = normalize_x(psi_0)
    dt = time_step
    psi_t = psi_0
    t = t0
    
    number_of_iterations = int(total_time/dt)
    
    for _ in range(number_of_iterations):   
        
        k1 = dt * dpsi_dt(t, psi_t)
        k2 = dt * dpsi_dt(t + dt/2, psi_t + k1/2)
        k3 = dt * dpsi_dt(t + dt/2, psi_t + k2/2)
        k4 = dt * dpsi_dt(t + dt, psi_t + k3)

        psi_t = psi_t + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
        
    return psi_t

# %%
#time_evolved_wavefunction_rk4 = wavefunction_t(final_time)

#data1 = N_atom*np.abs(time_evolved_wavefunction_rk4)**2*dx

#np.save("wavefunction_rk4.npy",data1)


psi_0 = np.complex64(psi_initial_for_full_potential)
psi_0 = normalize_x(psi_0)
#dt = 1.e-4
psi_t = psi_0
t0 = 0.0
#t_eval = np.arange(t0, final_time, dt)
t_eval = np.linspace(t0,final_time,2)
sol = scipy.integrate.solve_ivp(dpsi_dt, 
                                t_span = [t0, final_time],
                                y0 = psi_0, 
                                t_eval = t_eval,
                                method="RK23")


data1 = N_atom*np.abs(sol.y[:,-1])**2*dx
np.save("wavefunction_rk4_scipy.npy",data1)
