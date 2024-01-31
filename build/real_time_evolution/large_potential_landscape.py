# %%
# import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse

# %% [markdown]
# ## Source well parameters

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

N_atom = 40000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)
a_0 = np.sqrt(H_BAR/(trap_frequency*M))
#x_s = a_0
x_s = (4*np.pi*a_s*N_atom*a_0**4)**(1/5)    
print(x_s)
epsilon = (H_BAR/(M*omega_x*x_s**2))
print(epsilon)
delta   = (g_source*N_atom*(x_s**2))/(a_0**3*H_BAR*omega_x)

# %% [markdown]
# ## Source well potential

# %%
# number of discretized intervals in the position array
N = 2**17 # choose a power of two to make things simpler for fast Fourier transform

# barrier height outside the trap.
infinite_barrier_height = 10**5*10**3*2*PI*H_BAR


position_start = -41*1.e-6/x_s # m
position_end   = 41*1.e-6/x_s # m

dx = (position_end-position_start)/N

source_well_bias_potential = 0*10**3*2*PI*H_BAR

source_well_start          = -40*1.e-6/x_s # m
source_well_end            =  40*1.e-6/x_s # m

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

# changing potential into dimensionless unit
potential_arr = potential_arr/(epsilon*M*omega_x**2*x_s**2)


# %% [markdown]
# ## time split code

# %%
# discretizing the momentum space
L  = (position_end-position_start)
dk = (2*PI)/L

# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
def Hamiltonian_x(potential_array,psi): # H(x)
    return potential_array/(epsilon*M*omega_x**2*x_s**2)+delta*epsilon**(3/2)*np.abs(psi)**2

# momentum space discretization.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
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

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 0.01
time_step_SI  = -1j*10**(-7)   
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
psi_ITE = time_split_suzukui_trotter(psi_initial,potential_arr,time_step,final_time);
#print("Normalization of the wavefucntion = ",np.sqrt(np.sum(np.abs(psi_ITE)**2)*dx) )

np.save("ground_state_source_well.npy",psi_ITE)
# %%

# %%


# %% [markdown]
# # transistor potential landscape

# %%
x_s

# %%
# height of the infinite barrier in the source and the drain well.
infinite_barrier_height = 10**6*10**3*2*PI*H_BAR

position_start = -41*1.e-6/x_s # this must be same as the position start used earlier in the source well
position_end   = 100*1.e-6/x_s # right end of the transistor potential

# dx must be constant throught the code
N = int((position_end-position_start)/dx) # number of points in the discretized position
print(" Number of divisions in position, N = ",N,"\n")

source_well_bias_potential = 0       # must be same as the source well used earlier
source_well_start          = -40*1.e-6/x_s # must be same as the source well used earlier
source_well_end            = 40*1.e-6/x_s # must be same as the source well used earlier

source_gate_barrier_start  = source_well_end
source_gate_barrier_end    = 44*1.e-6/x_s

gate_bias_potential        = 0
gate_well_start            = source_gate_barrier_end
gate_well_end              = 64*1.e-6/x_s

gate_drain_barrier_start   = gate_well_end
gate_drain_barrier_end     = 68*1.e-6/x_s

drain_well_start           = gate_drain_barrier_end
drain_well_end             = 98*1.e-6/x_s

SG_barrier_height = 32*10**3*2*PI*H_BAR
GD_barrier_height = 31*10**3*2*PI*H_BAR

# creates the position array as an equally spaced array.
source_gate_drain_well_position = np.linspace(position_start,position_end,N)
# start with a zero initialized position array of length N.
source_gate_drain_well_potential = np.zeros(N)

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

# making the potential dimensionless       
#source_gate_drain_well_potential = source_gate_drain_well_potential/(epsilon*M*omega_x**2*x_s**2)   

# %%


# %% [markdown]
# # time split to evolve the wavefunction in real time

# %%
# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    
time_lst_index = int(sys.argv[1])    
time_lst = np.linspace(0,160,16)

final_time_SI = time_lst[time_lst_index]*10**(-3)
time_step_SI  = 10**(-8)  
# time is made dimensionless  
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        source_gate_drain_well_potential,
                                        time_step,final_time)

np.save("time_evolved_full_potential_time_split.npy",time_evolved_wavefunction_time_split)

# %%
D2 = scipy.sparse.diags([1, -2, 1], 
                        [-1, 0, 1],
                        shape=(source_gate_drain_well_position.size, source_gate_drain_well_position.size)) / dx**2
# kinetic part of the Hamiltonian
Hamiltonian = - (epsilon/2) * D2

def dpsi_dt(t,psi):
    return -1j*(Hamiltonian.dot(psi) + 
                (source_gate_drain_well_potential/(epsilon*M*omega_x**2*x_s**2))*psi + 
                delta*epsilon**(3/2)*np.abs(psi)**2*psi)

psi_0 = np.complex64(psi_initial_for_full_potential)
psi_0 = normalize_x(psi_0)
#dt = 1.e-4
psi_t = psi_0
t0 = 0.0
#t_eval = np.arange(t0, final_time, dt)
t_eval = np.linspace(t0,final_time,10)
sol = scipy.integrate.solve_ivp(dpsi_dt, 
                                t_span = [t0, final_time],
                                y0 = psi_0, 
                                t_eval = t_eval,
                                method="RK45")
np.save("time_evolved_full_potential_rk45.npy",sol.y[:,-1])


# %% [markdown]
# ## 


