# %%
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
TRAP_FREQUENCY = 1000 # Hz
OMEGA_X = TRAP_FREQUENCY
TRAP_LENGTH = np.sqrt(H_BAR/(ATOM_MASS*TRAP_FREQUENCY)) # m
CROSS_SECTIONAL_AREA = PI*TRAP_LENGTH**2 # m*m

NUMBER_OF_ATOMS = 100000

# interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(CROSS_SECTIONAL_AREA*ATOM_MASS)
a_0 = np.sqrt(H_BAR/(TRAP_FREQUENCY*ATOM_MASS))
#print(4*PI*a_s*NUMBER_OF_ATOMS, "must be greater than ", np.sqrt(H_BAR/(TRAP_FREQUENCY*ATOM_MASS)))
x_s = (4*np.pi*a_s*NUMBER_OF_ATOMS*a_0**4)**(1/5)    
#print("x_s = ",x_s)
epsilon = (H_BAR/(ATOM_MASS*OMEGA_X*x_s**2))
#print("epsilon = ",epsilon)
delta   = (g_source*NUMBER_OF_ATOMS*(x_s**2))/(a_0**3*H_BAR*OMEGA_X)
#print("delta =", delta)

# %% [markdown]
# #### Transistor potential Gaussian barrier and harmonic gate well

# %%
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)

N = 2**12

source_well_bias_potential_lst = np.around(np.linspace(5,20,16),2)#[i for i in range(32)]
np.save("V_SS_lst.npy", source_well_bias_potential_lst)

source_well_bias_potential_index = int(sys.argv[1])
V_SS = source_well_bias_potential_lst[source_well_bias_potential_index]

V_INFINITE_BARRIER  = 3000

# Position parameters in micrometers.
position_start      = -25
position_end        = 150
source_well_start   = -20
gate_well_start     = 0
gate_well_end       = 4.8
drain_well_end      = 140

np.save("position_start.npy",position_start)
np.save("position_end.npy",position_end)
np.save("source_well_start.npy",source_well_start)
np.save("gate_well_start.npy",gate_well_start)
np.save("gate_well_end.npy",gate_well_end)
np.save("drain_well_end.npy",drain_well_end)

"""
This function creates a smooth potential barrier with zero potential on the right of a given position x_0.
"""
def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))
"""
This function creates a smooth potential barrier with zero potential on the left of a given position x_0.
"""
def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(smoothness_control_parameter))

"""
This function creates a Gaussian barrier centered at x_0 with a given barrier height with two different potential plateaus on the left and right of the barrier.
"""
def different_plataeu_gaussian_barrier(x, x_0, barrier_height, left_plataeu_height, right_plataeu_height, sigma_1, sigma_2):
     return (barrier_height - (left_plataeu_height+right_plataeu_height)/2)*np.exp(-((x-x_0)/sigma_1)**2) + (left_plataeu_height + right_plataeu_height)/2 - (left_plataeu_height - right_plataeu_height)/2 * np.tanh((x-x_0)/(barrier_height*sigma_2))

"""
This function returns the coefficients of a harmonic well equation given three points on the well.
"""
def harmonic_well(x1,y1,x2,y2,x3,y3):
     A = np.array([[x1**2, x1, 1],
                   [x2**2, x2, 1],
                   [x3**2, x3, 1]])
     b = np.array([y1, y2, y3])
     c1, c2, c3 = np.linalg.solve(A, b)
     return c1, c2, c3

def transistor_potential_landscape(position_arr, SG_barrier_height, GD_barrier_height,
     # These parameters control the width of the barriers and the smoothness of the transitions.
     sigma = 0.7,
     SIGMA_1 = 0.6,
     SIGMA_2 = 0.01,
     SIGMA_3 = 0.6,
     SIGMA_4 = 1.0,
     ):

     # Creating the source well with V_SS on the left and a zero potential on the right.
     potential = different_plataeu_gaussian_barrier(position_arr, gate_well_start, SG_barrier_height, V_SS, 0, SIGMA_1, SIGMA_2)

     # Creating the bias potential in the source well.
     potential = np.where(position_arr < (source_well_start+gate_well_start)/2, V_INFINITE_BARRIER/2 - V_INFINITE_BARRIER/2 * np.tanh((position_arr-source_well_start)/(1.8)) + V_SS, potential)        

     """
     Three points are chosen to create a harmonic well in the gate well region. The points are shifted by an amount
     delta_left and delta_right to create a smooth transition between the Gaussian barriers and the harmonic well.
     """
     delta_left = 0.0
     delta_right = 0.15
     x_1 = gate_well_start + delta_left
     y_1 = different_plataeu_gaussian_barrier(gate_well_start + delta_left, gate_well_start, SG_barrier_height, V_SS, 0, SIGMA_1, SIGMA_2)
     x_2 = gate_well_end - delta_right
     y_2 = different_plataeu_gaussian_barrier(gate_well_end - delta_right, gate_well_end, GD_barrier_height, 0,0, SIGMA_3, SIGMA_4)
     x_3 = (gate_well_start + gate_well_end)/2
     y_3 = 2

     # Coefficients of the harmonic well equation.
     pp, qq, rr = harmonic_well(x_1, y_1, x_2, y_2, x_3, y_3)

     # This loop creates the harmonic well in the gate well region and the GD barrier.
     for i in range(len(position_arr)):
          if position_arr[i] >= gate_well_start + delta_left and position_arr[i] <= gate_well_end - delta_right:
               potential[i] = pp*position_arr[i]**2 + qq*position_arr[i] + rr
          elif position_arr[i] >= gate_well_end - delta_right:
               potential[i] = different_plataeu_gaussian_barrier(position_arr, gate_well_end, GD_barrier_height, 0,0, SIGMA_3, SIGMA_4)[i]

     # Creates a barrier at the end of the drain well.
     potential = np.where(position_arr > (gate_well_end+drain_well_end)/2, right_tanh_function(position_arr, V_INFINITE_BARRIER, drain_well_end, 0.5), potential)
     return potential

# fig, ax = plt.subplots()
# fig.set_figwidth(12)
# fig.set_figheight(5)
xs = np.linspace(position_start,position_end,N)
# changing position to SI units
xs_SI = xs*1.e-6
# changing potential to SI units
barrier_width_control_parameter = 0.9 # change this to control the width of the barriers
complete_transistor_potential = transistor_potential_landscape(xs,30,32,barrier_width_control_parameter)
complete_transistor_potential_SI = complete_transistor_potential*10**3*2*PI*H_BAR
np.save("complete_transistor_position_SI.npy", xs_SI)
np.save("complete_transistor_potential_SI.npy", complete_transistor_potential_SI)

# plt.plot(xs_SI*1.e6, complete_transistor_potential_SI/(10**3*2*PI*H_BAR), linewidth = 2, color = "k")
# #plt.axhline(y=V_SS, color="k", linestyle='--')   
# plt.axhline(y=0, color="k", linestyle='--')
# plt.axvline(x=source_well_start, color="k", linestyle='--')
# plt.axvline(x=gate_well_start, color="k", linestyle='--')
# plt.axvline(x=gate_well_end, color="k", linestyle='--')
# plt.xlim([-20, 25])
# plt.ylim([0,31*1.2])
# plt.ylabel(r"Potential, $V(x),\; (kHz)$",labelpad=10)
# plt.xlabel(r"Position, $x, \; (\mu m)$",labelpad=10)
# fig.tight_layout(pad=1.0)
# path = "/Users/sasankadowarah/atomtronics/"
# os.chdir(path)
# np.save("transistor_position_gaussian.npy",  xs_SI)
# np.save("transistor_potential_gaussian.npy", complete_transistor_potential_SI)
# plt.savefig("complete_transistor_potential_harmonic_gate_well.png", dpi=60)
#plt.show()

# %% [markdown]
# #### Source well potential

# %%
complete_transistor_position = xs_SI/x_s
source_well_position = complete_transistor_position[np.where((complete_transistor_position > position_start/x_s) & (complete_transistor_position < gate_well_start/x_s))]

np.save("source_well_position.npy", source_well_position)
# %%
source_well_potential = complete_transistor_potential_SI[0:len(source_well_position)]
np.save("source_well_potential.npy", source_well_potential)

# %%
dx = (complete_transistor_position[-1] - complete_transistor_position[0])/len(complete_transistor_position)
#print("dx (SI)= ", dx*x_s)  

# f = plt.figure()    
# plt.plot(source_well_position, source_well_potential,linewidth=3)
# plt.ylim([0,33*10**3*2*PI*H_BAR*1.02])
# ax = f.gca()
# #ax.axhline(0, color="green",linestyle="--",linewidth = 2)
# plt.xlabel(r"Dimentionless position, $\tilde{x}$", labelpad = 20)
# plt.ylabel(r"Potential, $V_{\textrm{Source}}$", labelpad = 20)
# f.set_figwidth(10)
# f.set_figheight(6)
# plt.show()

# %% [markdown]
# #### time split code

# %%
L  = np.abs((source_well_position[-1]-source_well_position[0]))
N = len(source_well_position)
dx = L/N

#print("dx (SI) = ", dx*x_s)
# discretizing the momentum space

dk = (2*PI)/L
#print("dk = ", dk)

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
                    path = "/Users/sasankadowarah/atomtronics/cluster-codes/harmonic_gate_well/jupyter_notebook_data/"
                    os.chdir(path)
                    snapshots_lst.remove(snapshot_time) # removing the used time to avoid multiple entries
                    np.save("time_evolved_wavefunction_"+str(np.around(time,3))+".npy",psi_x)    

    psi_x = normalize_x(psi_x) # returns the normalized wavefunction
    
    return psi_x

# %% [markdown]
# #### imaginary time evolution for ground state in the source well

# %%
# start with an initial state
# psi_initial = np.ones(N)
# psi_initial = normalize_x(psi_initial) 

psi_initial = np.exp(-(source_well_position+1.5)**2/(0.1)) #np.ones(N)
psi_initial = normalize_x(psi_initial)

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 1.e-2
time_step_SI  = -1j*10**(-6)   
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time, [])
#print("Normalization of the wavefunction = ",np.sqrt(np.sum(np.abs(psi_source_well_ITE)**2)*dx) )
np.save("source_well_ground_state.npy", psi_source_well_ITE)

# %%
# psi_initial = np.exp(-(source_well_position+1)**2/(0.1)) #np.ones(N)
# psi_initial = normalize_x(psi_initial) 
# plt.plot(source_well_position, psi_initial,linewidth=3)
# plt.plot(source_well_position,source_well_potential/(np.max(source_well_potential)),linewidth=3)
# plt.show()

# %% [markdown]
# #### plot ground state wavefunction  in the source well

# %%
data0 = source_well_position
data1 = np.abs(psi_source_well_ITE)**2*dx
data3 = source_well_potential#/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)

# fig, ax1 = plt.subplots()

# ax1.set_xlabel(r"Position, $\tilde{x}$", labelpad=20)
# ax1.set_ylabel(r"$|\tilde{\psi}|^{2}$", color="tab:red", labelpad=20)
# ax1.plot(data0, data1, color="tab:red",linewidth = 5)
# plt.title(r"Ground state wavefunction in the source well")
# #plt.legend()
# ax1.tick_params(axis="y", labelcolor="tab:red")
# ax2 = ax1.twinx()

# color = "tab:blue"
# ax2.set_ylabel(r"$V(x)/(\epsilon \omega^{2}_{x} x^{2}_{s})$ ", color=color,  labelpad=20)
# ax2.plot(data0, data3, color=color,linewidth = 5)
# #ax1.set_ylim([0,0.005])
# #ax1.set_xlim([-3,0.1])
# ax2.set_ylim([0,33*10**3*2*PI*H_BAR*1.02])
# ax2.tick_params(axis="y", labelcolor=color)
# fig.set_figwidth(12)
# fig.set_figheight(6)
# plt.subplots_adjust(bottom=0.2)
# for spine in ax1.spines.values():
#     spine.set_linewidth(2)
# ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
# ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
# ax2.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
# ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
# #plt.savefig("ground_state_in_source_well_8"+str(NUMBER_OF_ATOMS)+".jpg", dpi=300)
# fig.tight_layout()

# %%
# ground_state_in_source_well_k_space = fftpack.fft(psi_source_well_ITE)
# initial_state_in_k_space = fftpack.fft(psi_initial)
# plt.plot(np.abs(ground_state_in_source_well_k_space)**2*dk, linewidth = 2, label = "$\psi(k)$")
# #plt.plot(np.abs(initial_state_in_k_space)**2*dk, linewidth = 2,label = "Initial state")
# plt.legend()
# plt.show()

# %% [markdown]
# #### chemical potential of the BEC

# %%
data0 = source_well_position
data1 = source_well_potential +g_source*NUMBER_OF_ATOMS*np.abs(psi_source_well_ITE/np.sqrt(x_s))**2
data3 = source_well_potential 

fig, ax1 = plt.subplots()
ax1.set_xlabel(r"Position, $\tilde{x}$", labelpad=20)
ax1.set_ylabel(r"Chemical potential $\mu\; (J)$", color="tab:red", labelpad=20)
ax1.plot(data0, data1, color="tab:red",linewidth = 5)
plt.title(r"Chemical potential in the source well")
#plt.legend()
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel(r"$V(x)\; (J)$ ", color=color,  labelpad=20)
ax2.plot(data0, data3, color=color,linewidth = 5)
ax2.tick_params(axis="y", labelcolor=color)
fig.set_figwidth(12)
fig.set_figheight(6)
plt.subplots_adjust(bottom=0.2)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
ax2.set_ylim([0, 30 * 10**3*2*PI*H_BAR*1.2 ])
ax1.set_ylim([0, 33 * 10**3*2*PI*H_BAR*1.2 ])
ax1.axhline(y = 33 * 10**3*2*PI*H_BAR , color="k", linestyle='--')
ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
#ax1.set_xlim([-4,0])
plt.savefig("chemical_potential_in_source_well.jpg", dpi=300)
fig.tight_layout()

# %% [markdown]
# # time split for real time evolution

# %%
N = len(complete_transistor_position)
L = np.abs((complete_transistor_position[-1]-complete_transistor_position[0]))
dx = L/N

# We do not have to non-dimensionalize dk since transistor position and hence L is already non-dimensional.
dk = (2*PI)/L
#print("dx = ", dx*x_s)
#print("dk = ", dk)
# Momentum space discretization done for periodic boundary conditions.
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# Put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    

final_time_SI = 30*10**(-3)
time_step_SI  = 10**(-8)  

# Time is made dimensionless.  
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
time_lst = list(np.arange(0,int(final_time_SI*1.e3),0.5))
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential_SI,
                                        time_step, final_time, time_lst)