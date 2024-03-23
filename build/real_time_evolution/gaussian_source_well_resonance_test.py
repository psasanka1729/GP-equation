# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
import scipy.sparse

# %% [markdown]
# #### Transistor parameters and constants

# %%
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)

r""" Rb87 parameters """
ATOM_MASS   = 1.4192261*10**(-25) # kg
a_s = 98.006*5.29*10**(-11) # m https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.053614
TRAP_FREQUENCY = 918 # Hz
OMEGA_X = TRAP_FREQUENCY
TRAP_LENGTH = np.sqrt(H_BAR/(ATOM_MASS*TRAP_FREQUENCY)) # m
CROSS_SECTIONAL_AREA = PI*TRAP_LENGTH**2 # m*m

NUMBER_OF_ATOMS = 20000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(CROSS_SECTIONAL_AREA*ATOM_MASS)
a_0 = np.sqrt(H_BAR/(TRAP_FREQUENCY*ATOM_MASS))
#x_s = a_0
x_s = (4*np.pi*a_s*NUMBER_OF_ATOMS*a_0**4)**(1/5)    
#print(x_s)
epsilon = (H_BAR/(ATOM_MASS*OMEGA_X*x_s**2))
#print(epsilon)
delta   = (g_source*NUMBER_OF_ATOMS*(x_s**2))/(a_0**3*H_BAR*OMEGA_X)
#print(delta)

# %% [markdown]
# #### Transistor potential Gaussian barrier

# %%
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)
source_well_bias_potential_lst = [i for i in range(32)]
source_well_bias_potential_index = int(sys.argv[1])
V_SS = source_well_bias_potential_lst[source_well_bias_potential_index]  # kHz
V_INFINITE_BARRIER  = 200
N = 2**12

np.save("N.npy", N)

position_start      = -30
position_end        = 60
# positions in SI units
source_well_start   = -10
gate_well_start     = 0
gate_well_end       = 6
drain_well_end      = 40

def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

def gaussian_barrier(x, mu, SG_barrier_height, GD_barrier_height, sigma):

     gaussian = (SG_barrier_height*np.exp(-((x-gate_well_start)/sigma)**2) +
                 GD_barrier_height*np.exp(-((x-gate_well_end)/sigma)**2))

     plataeu_x_cor =  (mu - sigma*np.sqrt(-np.log(V_SS/SG_barrier_height)))

     # bias potential in the source well
     gaussian = np.where(x < plataeu_x_cor, V_INFINITE_BARRIER/2 - V_INFINITE_BARRIER/2 * np.tanh((xs-source_well_start)/(0.001*V_INFINITE_BARRIER)) + V_SS, gaussian)   

     # infinite barrier
     gaussian = np.where(x < -20, V_INFINITE_BARRIER, gaussian)

     gaussian = np.where(x > (gate_well_end+drain_well_end)/2, right_tanh_function(xs, V_INFINITE_BARRIER, drain_well_end, 0.005), gaussian)
     return gaussian

xs = np.linspace(position_start,position_end,N)
xs_SI = xs*1.e-6
complete_transistor_position = xs_SI/x_s
complete_transistor_potential = gaussian_barrier(xs,0,30,33,1.27)

np.save("transistor_position.npy", complete_transistor_position)
np.save("transistor_potential.npy", complete_transistor_potential)

source_well_position = complete_transistor_position[np.where((complete_transistor_position > position_start) & (complete_transistor_position < gate_well_start))]
source_well_potential = complete_transistor_potential[0:len(source_well_position)]

np.save("source_well_position.npy", source_well_position)
np.save("source_well_potential.npy", source_well_potential)

dx = (position_end-position_start)/N
N = len(source_well_position)

# discretizing the momentum space
L  = np.abs((source_well_position[-1]-source_well_position[0]))
dx = L/N

np.save("dx1.npy", dx)

dk = (2*PI)/L
np.save("dk.npy", dk)

# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
def Hamiltonian_x(potential_array_SI,psi): # H(x)
    return potential_array_SI/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)+delta*epsilon**(3/2)*np.abs(psi)**2

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

    
def time_split_suzukui_trotter(initial_wavefunction, potential_array, dt, total_time, snapshots_lst):    

    psi_k = fftpack.fft(initial_wavefunction)
    psi_x = initial_wavefunction
    
    total_iterations = int(np.abs(total_time)/np.abs(dt))
    #print("Number of iterations =", total_iterations)
    
    for iteration in range(total_iterations):

          time = (dt*iteration/OMEGA_X)*1.e3 # changing to SI units miliseconds for convenience
     
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
    
          if snapshots_lst: 
               for snapshot_time in snapshots_lst:
                    if np.abs(np.around(time,3) - snapshot_time) < 1.e-10:
                         snapshots_lst.remove(snapshot_time) # removing the used time to avoid multiple entries
                         np.save("time_evolved_wavefunction_"+str(np.around(time,3))+".npy",psi_x)

    psi_x = normalize_x(psi_x) # returns the normalized wavefunction
    
    return psi_x

# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI  = 1.e-3
time_step_SI   = -1j*10**(-7)   
final_time     = OMEGA_X*final_time_SI
time_step      = OMEGA_X*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time, [])

np.save("ground_state_in_source_well.npy", psi_source_well_ITE)

N = len(complete_transistor_position)
# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE

while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    
final_time_SI  = 20*10**(-3)
time_step_SI   = 10**(-7)  
# time is made dimensionless  
final_time     = OMEGA_X*final_time_SI
time_step      = OMEGA_X*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential*10**3*2*PI*H_BAR,
                                        time_step,final_time,[t for t in range(20)])
np.save("dx2.npy", dx)                                        
