# %%
# import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.sparse

# %%

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

N_atom = 4000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(A*M)
a_0 = np.sqrt(H_BAR/(trap_frequency*M))
#x_s = a_0
x_s = (4*np.pi*a_s*N_atom*a_0**4)**(1/5)    
print(x_s)
epsilon = (H_BAR/(M*omega_x*x_s**2))
print(epsilon)
delta   = (g_source*N_atom*(x_s**2))/(a_0**3*H_BAR*omega_x)
print(delta)

# %%
r"""

Returns f(x) = V_{0} * tanh((x-b)/a)

"""

def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(smoothness_control_parameter*barrier_height))

def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(smoothness_control_parameter*barrier_height))  

def smooth_potential_well(position_array,
                        source_well_start,
                        SG_barrier_start,
                        SG_barrier_end,
                        GD_barrier_start,
                        GD_barrier_end,
                        drain_well_end,
                        Kappa):

        infinite_barrier = 1000
        SG_barrier_height = 31
        GD_barrier_height = 32    
        source_well_bias_potential = 20
        potential_well_arr = np.zeros(len(position_array))

        position_start = position_array[0]
        position_end   = position_array[-1]

        for pos in range(len(position_array)):

                position = position_array[pos]
                if position_start <= position <= (source_well_start + SG_barrier_start)/2:
                        # infinite barrier on the left side
                        potential_well_arr[pos] = left_tanh_function(position, infinite_barrier, source_well_start, Kappa) 

                elif (source_well_start + SG_barrier_start)/2 <= position <= (SG_barrier_start + SG_barrier_end)/2:
                        # left side of SG barrier 
                        potential_well_arr[pos] = right_tanh_function(position, SG_barrier_height, SG_barrier_start, Kappa)

                elif (SG_barrier_start + SG_barrier_end)/2 <= position <= (SG_barrier_end + GD_barrier_start)/2:
                        # right side of SG barrier
                        potential_well_arr[pos] = left_tanh_function(position, SG_barrier_height, SG_barrier_end, Kappa)

                elif (SG_barrier_end + GD_barrier_start)/2 <= position <= (GD_barrier_start + GD_barrier_end)/2:
                        # left side of GD barrier
                        potential_well_arr[pos] = right_tanh_function(position, GD_barrier_height, GD_barrier_start, Kappa)

                elif (GD_barrier_start + GD_barrier_end)/2 <= position <= (GD_barrier_end + drain_well_end)/2:
                        # right side of GD barrier
                        potential_well_arr[pos] = left_tanh_function(position, GD_barrier_height, GD_barrier_end, Kappa)

                elif (GD_barrier_end + drain_well_end)/2 <= position <= position_end:
                        # infinite barrier on the right side
                        potential_well_arr[pos] = right_tanh_function(position, infinite_barrier, drain_well_end, Kappa)

        for pos in range(len(position_array)):
                position = position_array[pos]
                if source_well_start <= position < SG_barrier_start:
                        potential_well_arr[pos] += source_well_bias_potential
                else:
                        pass

        return potential_well_arr*10**3*2*PI*H_BAR

N = 2**12
position_start = -50
position_end   = 100
complete_transistor_position  = np.linspace(position_start, position_end, N)*1.e-6/x_s

# positions in dimensionless form
source_well_start = -40*1.e-6/x_s 
SG_barrier_start = -3*1.e-6/x_s
SG_barrier_end = 3*1.e-6/x_s
GD_barrier_start = 15*1.e-6/x_s
GD_barrier_end = 21*1.e-6/x_s
drain_well_end= 70*1.e-6/x_s

complete_transistor_potential = smooth_potential_well(complete_transistor_position, source_well_start, SG_barrier_start, SG_barrier_end, GD_barrier_start,GD_barrier_end, drain_well_end, 0.0005)



# %% [markdown]
# #### Source well potential

# %%
source_well_position = complete_transistor_position[np.where((complete_transistor_position > position_start) & (complete_transistor_position < SG_barrier_start))]
source_well_potential = np.zeros(len(source_well_position))

Kappa = 0.0005
infinite_barrier = 1000
SG_barrier_height = 30
for pos in range(len(source_well_position)):

        position = source_well_position[pos]
        if position_start <= position <= (source_well_start + SG_barrier_start)/2:
                # infinite barrier on the left side
                source_well_potential[pos] = left_tanh_function(position, infinite_barrier, source_well_start, Kappa)

        else:
                # left side of SG barrier 
                source_well_potential[pos] = right_tanh_function(position, infinite_barrier, SG_barrier_start, Kappa)

source_well_potential = source_well_potential*10**3*2*PI*H_BAR         

np.save("source_well_position_dimensionless.npy", source_well_position)
np.save("source_well_potential_SI.npy", source_well_potential)

dx = (position_end-position_start)/N


# %% [markdown]
# #### time split code

# %%
N = len(source_well_position)
print("N = ",N)

# discretizing the momentum space
L  = np.abs((source_well_position[-1]-source_well_position[0]))

dx = L/N
print("dx (SI) = ", dx*x_s)

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

# %%
# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 1.e-2
time_step_SI  = -1j*10**(-7)   
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time);

np.save("ground_state_in_source_well.npy", psi_source_well_ITE)


# %% [markdown]
# # time split to evolve the wavefunction in real time

# %%
N = len(complete_transistor_position)
# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    


time_lst = np.linspace(0,1000,16)    
time_index = int(sys.argv[1])

final_time_SI = time_lst[time_index]*10**(-3)
time_step_SI  = 10**(-7)  
# time is made dimensionless  
final_time = omega_x*final_time_SI
time_step = omega_x*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential,
                                        time_step,final_time)



# %%
data0 = complete_transistor_position
data1 = N_atom*np.abs(time_evolved_wavefunction_time_split)**2*dx
data2 = np.abs(time_evolved_wavefunction_time_split)**2
data3 = complete_transistor_potential/(epsilon*M*omega_x**2*x_s**2)

np.save("complete_transistor_position.npy", data0)
np.save("real_time_evolved_wavefunction_time_split.npy", data2)
np.save("complete_transistor_potential.npy", data3)


# %% [markdown]
# # Runge Kutta algorithm for time evolution

# %% [markdown]
# ## Custom RK4

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


# %% [markdown]
# ## RK4 of scipy

# %%
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
                                method="RK45")



# %%
data1 = N_atom*np.abs(sol.y[:,-1])**2*dx
np.save("wavefunction_rk4_scipy.npy",data1)



