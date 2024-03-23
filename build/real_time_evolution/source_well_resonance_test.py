# %%
# import os
import sys
import numpy as np
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
np.save("x_s.npy", x_s)
#print(x_s)
epsilon = (H_BAR/(ATOM_MASS*OMEGA_X*x_s**2))
np.save("epsilon.npy",epsilon)
#print(epsilon)
delta   = (g_source*NUMBER_OF_ATOMS*(x_s**2))/(a_0**3*H_BAR*OMEGA_X)
np.save("delta.npy",delta)
#print(delta)

# %% [markdown]
# #### Complete transistor potential

# %%
r"""
Returns f(x) = V_{0} * tanh((x-b)/a)
"""

def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))  

infinite_barrier = 500  # kHz
SG_barrier_height_kHz = 30   # kHz
GD_barrier_height_kHz = 32   # kHz
source_well_bias_potential_lst = [i for i in range(32)]
source_well_bias_potential_index = int(sys.argv[1])
source_well_bias_potential = source_well_bias_potential_lst[source_well_bias_potential_index]  # kHz
gate_well_bias_potential   = 0 # kHz
def smooth_potential_well(position_array,
                        source_well_start,
                        SG_barrier_start,
                        SG_barrier_height,
                        SG_barrier_end,
                        GD_barrier_start,
                        GD_barrier_height,
                        GD_barrier_end,
                        drain_well_end,
                        Kappa):

        potential_well_arr = np.zeros(len(position_array))

        position_start = position_array[0]
        position_end   = position_array[-1]

        for pos in range(len(position_array)):

                position = position_array[pos]

                if position_start <= position <= (source_well_start + SG_barrier_start)/2:
                        # infinite barrier on the left side
                        potential_well_arr[pos] =  infinite_barrier/2 - infinite_barrier/2 * np.tanh((position-source_well_start)/(INFINITE_STEP_SMOOTHNESS*infinite_barrier)) + source_well_bias_potential

                elif (source_well_start + SG_barrier_start)/2 <= position <= (SG_barrier_start + SG_barrier_end)/2:
                        
                        # left side of SG barrier                              
                        potential_well_arr[pos] = (SG_barrier_height+source_well_bias_potential)/2 + (SG_barrier_height-source_well_bias_potential)/2 * np.tanh((position-SG_barrier_start)/(Kappa*SG_barrier_height))

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
                        potential_well_arr[pos] = right_tanh_function(position, infinite_barrier, drain_well_end, INFINITE_STEP_SMOOTHNESS)

        return potential_well_arr*10**3*2*PI*H_BAR # Joule

N = 2**12
position_start = -30*1.e-6/x_s # micrometer
position_end   = 60*1.e-6/x_s # micrometer

complete_transistor_position  = np.linspace(position_start, position_end, N)

INFINITE_STEP_SMOOTHNESS = 0.0001
FINITE_STEP_SMOOTHNESS = 0.005 # less will give sharper transition

# positions in dimensionless form
source_well_start = -20*1.e-6/x_s 
SG_barrier_start = -3*1.e-6/x_s
SG_barrier_end =    0*1.e-6/x_s
GD_barrier_start = 3*1.e-6/x_s
GD_barrier_end = 6*1.e-6/x_s
drain_well_end = 50*1.e-6/x_s

complete_transistor_potential = smooth_potential_well(complete_transistor_position, source_well_start, SG_barrier_start, SG_barrier_height_kHz, SG_barrier_end, GD_barrier_start,GD_barrier_height_kHz, GD_barrier_end, drain_well_end, FINITE_STEP_SMOOTHNESS)

np.save("transistor_position.npy", complete_transistor_position)
np.save("transistor_potential.npy", complete_transistor_potential)
# %% [markdown]
# #### Source well potential

# %%
source_well_position = complete_transistor_position[np.where((complete_transistor_position > position_start) & (complete_transistor_position < SG_barrier_end))]
source_well_potential = np.zeros(len(source_well_position))

SMOOTHNESS = 0.0005
for pos in range(len(source_well_position)):

        position = source_well_position[pos]
        if position_start <= position <= (source_well_start + SG_barrier_start)/2:
                # infinite barrier on the left side
                source_well_potential[pos] = left_tanh_function(position, infinite_barrier, source_well_start, SMOOTHNESS)

        else:
                # left side of SG barrier 
                source_well_potential[pos] = right_tanh_function(position, infinite_barrier, SG_barrier_start, SMOOTHNESS)

# converting the potential to SI units
source_well_potential = (source_well_potential+source_well_bias_potential)*10**3*2*PI*H_BAR         

np.save("source_well_position.npy", source_well_position)
np.save("source_well_potential.npy", source_well_potential)

dx = (position_end-position_start)/N
#print("dx (SI)= ", dx*x_s)  

# %% [markdown]
# #### time split code

# %%
N = len(source_well_position)
#print("N = ",N)

# discretizing the momentum space
L  = np.abs((source_well_position[-1]-source_well_position[0]))

dx = L/N

np.save("dx.npy", dx)
#print("dx (SI) = ", dx*x_s)

dk = (2*PI)/L

# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
def Hamiltonian_x(potential_array,psi): # H(x)
    return potential_array/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)+delta*epsilon**(3/2)*np.abs(psi)**2

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

    
def time_split_suzukui_trotter(initial_wavefunction,potential_array,dt,total_time,snapshots_lst):    

    psi_k = fftpack.fft(initial_wavefunction)
    psi_x = initial_wavefunction
    
    total_iterations = int(np.abs(total_time)/np.abs(dt))
    print("Number of iterations =", total_iterations)
    
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

# %% [markdown]
# #### imaginary time evolution for ground state in the source well

# %%
# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 1.e-3
time_step_SI  = -1j*10**(-7)   
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time,[]);
#print("Normalization of the wavefucntion = ",np.sqrt(np.sum(np.abs(psi_source_well_ITE)**2)*dx) )

# %% [markdown]
# #### plot ground state wavefunction  in the source well

# %%
data0 = source_well_position
data1 = np.abs(psi_source_well_ITE)**2*dx
data3 = source_well_potential/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)

np.save("ground_state_in_source_well.npy", psi_source_well_ITE)

# %% [markdown]
# #### chemical potential of the BEC

# %%
source_well_position_to_plot = complete_transistor_position[0:len(source_well_potential)]
source_well_potential_to_plot = complete_transistor_potential[0:len(source_well_potential)]

data0 = source_well_position_to_plot
data1 = source_well_potential +g_source*NUMBER_OF_ATOMS*np.abs(psi_source_well_ITE/np.sqrt(x_s))**2
data3 = source_well_potential 

# %% [markdown]
# ### real time evolution in the complete transistor potential

# %% [markdown]
# #### time split for real time evolution

# %%
N = len(complete_transistor_position)
# momentum space discretization
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    
final_time_SI = 20*10**(-3)
time_step_SI  = 10**(-7)  
# time is made dimensionless  
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential,
                                        time_step,final_time,[t for t in range(20)])

# %%
# plotting everything in SI units
data0 = complete_transistor_position*x_s*1.e6
data1 = np.abs(time_evolved_wavefunction_time_split)**2*dx
#data2 = np.abs(time_evolved_wavefunction_time_split)**2
data3 = complete_transistor_potential#/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)

#np.save("time_evolved_wavefunction_"+str(time)+".npy", time_evolved_wavefunction_time_split)

# %%
#print("Total number of atoms in the trap = ", np.sum(NUMBER_OF_ATOMS*np.abs(time_evolved_wavefunction_time_split)**2*dx))  

# %%
# atom number in source well 805 (approx)
# atom number in gate well 111
# drain 186
# SG = 21
# GD = 23
# 62 (ms)
# time to evolve = 10 (ms)

# %% [markdown]
# #### Runge Kutta algorithm for time evolution

# %% [markdown]
# #### Custom RK4

# %%
# N = len(complete_transistor_position)
# # momentum space discretization
# k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
# E_k = k**2*epsilon/2

# # put the initial ground state in the source well of the transistor.
# psi_initial_for_full_potential = psi_source_well_ITE

# while len(psi_initial_for_full_potential) < N:
#     psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
# D2 = scipy.sparse.diags([1, -2, 1], 
#                         [-1, 0, 1],
#                         shape=(complete_transistor_position.size, complete_transistor_position.size)) / (dx)**2

# # kinetic part of the Hamiltonian
# Hamiltonian = - (epsilon/2) * D2

# def dpsi_dt(t,psi):
#     return -1j*(Hamiltonian.dot(psi) + 
#                 (complete_transistor_potential/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2))*psi + 
#                 delta*epsilon**(3/2)*np.abs(psi)**2*psi)

# %% [markdown]
# #### RK4 of scipy

# %%
# psi_0 = np.complex64(psi_initial_for_full_potential)
# psi_0 = normalize_x(psi_0)
# psi_t = psi_0
# t0 = 0.0
# #t_eval = np.arange(t0, final_time, dt)
# t_eval = np.linspace(t0,final_time,2)
# # sol = scipy.integrate.solve_ivp(dpsi_dt, 
# #                                 t_span = [t0, final_time],
# #                                 y0 = psi_0, 
# #                                 t_eval = t_eval,
# #                                 method="RK45")

# %%
# t0 = 0.0
# time_step_SI  = 10**(-7)    
# time_step = OMEGA_X*time_step_SI
# def custom_rk4_time_evolution(total_time):
#     # initial wavefunction
#     psi_0 = np.complex64(psi_initial_for_full_potential)
#     psi_0 = normalize_x(psi_0)
#     dt = time_step
#     psi_t = psi_0
#     t = t0
    
#     number_of_iterations = int(total_time/dt)
#     print("Number of iterations = ", number_of_iterations)
#     for _ in range(number_of_iterations):   
        
#         k1 = dt * dpsi_dt(t, psi_t)
#         k2 = dt * dpsi_dt(t + dt/2, psi_t + k1/2)
#         k3 = dt * dpsi_dt(t + dt/2, psi_t + k2/2)
#         k4 = dt * dpsi_dt(t + dt, psi_t + k3)

#         psi_t = psi_t + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
#         t = t + dt
        
#     return psi_t

# #time_evolved_wavefunction_rk4 = custom_rk4_time_evolution(final_time)

# %%
# data0 = complete_transistor_position
# #data1 = np.abs(sol.y[:,-1])**2*dx
# data2 = np.abs(time_evolved_wavefunction_rk4**2)*dx
# data3 = complete_transistor_potential/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)
# data4 = np.abs(time_evolved_wavefunction_time_split)**2*dx
# fig, ax1 = plt.subplots()

# ax1.set_xlabel(r"Position, $\tilde{x}$")
# ax1.set_ylabel(r"$|\tilde{\psi}|^{2}$", color="tab:red")
# #ax1.scatter(t, data1, color=color,s= 5,linewidth = 3,label=r"Time = ")
# #ax1.plot(data0, data2, color="tab:red",linewidth = 3,label=r"Time split")
# #ax1.plot(data0, data1, color="tab:red",linewidth = 2,label=r" RK4 Scipy")
# ax1.plot(data0, data2, color="tab:orange",linewidth = 2,label=r"Custom RK4")
# ax1.plot(data0, data4, color="tab:green",linewidth = 2,label=r" Time split Time evolved wavefunction")
# plt.legend()
# #ax1.set_xlim([-6, 15])
# ax1.set_ylim([0, np.max(data1)*1.1])
# ax1.tick_params(axis='y', labelcolor="tab:red")
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = "tab:blue"
# ax2.set_ylabel(r"$V(x)/(\epsilon \omega^{2}_{x} x^{2}_{s})$ ", color=color)  # we already handled the x-label with ax1
# ax2.plot(data0, data3, color=color,linewidth = 5)
# ax2.set_ylim([0, (GD_barrier_height_kHz*10**3*2*PI*H_BAR)/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)*1.02])
# ax2.tick_params(axis="y", labelcolor=color)
# fig.set_figwidth(20)
# fig.set_figheight(7)
# plt.subplots_adjust(bottom=0.2)
# for spine in ax1.spines.values():
#     spine.set_linewidth(3)
# ax1.tick_params(axis="x", direction="inout", length=20, width=2, color="k")
# ax1.tick_params(axis="y", direction="inout", length=20, width=2, color="k")
# #plt.savefig("psi_"+str(NUMBER_OF_ATOMS)+"_"+str(i)+".jpg", dpi=600)
# fig.tight_layout()

# %% [markdown]
# ## 


