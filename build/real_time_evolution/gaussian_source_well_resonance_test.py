# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
import scipy.sparse

# %%
# matplotlib parameters 
large = 40; med = 40; small = 20
params = {'axes.titlesize': med,
          'axes.titlepad' : med,
          'legend.fontsize': med,
          'axes.labelsize': med ,
          'axes.titlesize': med ,
          'xtick.labelsize': med ,
          'ytick.labelsize': med ,
          'figure.titlesize': med}
#plt.rcParams['text.usetex'] = True # need LaTeX. Change it to False if LaTeX is not installed in the system
plt.rcParams.update(params)

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
# #### Complete transistor potential

# %%
# r"""
# Returns f(x) = V_{0} * tanh((x-b)/a)
# """

# def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
#                 return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

# def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
#                 return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))  

# infinite_barrier = 500  # kHz
# SG_barrier_height_kHz = 30   # kHz
# GD_barrier_height_kHz = 32   # kHz
# source_well_bias_potential = 26 # kHz
# gate_well_bias_potential   = 0 # kHz
# def smooth_potential_well(position_array,
#                         source_well_start,
#                         SG_barrier_start,
#                         SG_barrier_height,
#                         SG_barrier_end,
#                         GD_barrier_start,
#                         GD_barrier_height,
#                         GD_barrier_end,
#                         drain_well_end,
#                         Kappa):

#         potential_well_arr = np.zeros(len(position_array))

#         position_start = position_array[0]
#         position_end   = position_array[-1]

#         for pos in range(len(position_array)):

#                 position = position_array[pos]

#                 if position_start <= position <= (source_well_start + SG_barrier_start)/2:
#                         # infinite barrier on the left side
#                         potential_well_arr[pos] =  infinite_barrier/2 - infinite_barrier/2 * np.tanh((position-source_well_start)/(INFINITE_STEP_SMOOTHNESS*infinite_barrier)) + source_well_bias_potential

#                 elif (source_well_start + SG_barrier_start)/2 <= position <= (SG_barrier_start + SG_barrier_end)/2:
                        
#                         # left side of SG barrier                              
#                         potential_well_arr[pos] = (SG_barrier_height+source_well_bias_potential)/2 + (SG_barrier_height-source_well_bias_potential)/2 * np.tanh((position-SG_barrier_start)/(Kappa*SG_barrier_height))

#                 elif (SG_barrier_start + SG_barrier_end)/2 <= position <= (SG_barrier_end + GD_barrier_start)/2:
#                         # right side of SG barrier
#                         potential_well_arr[pos] = left_tanh_function(position, SG_barrier_height, SG_barrier_end, Kappa)

#                 elif (SG_barrier_end + GD_barrier_start)/2 <= position <= (GD_barrier_start + GD_barrier_end)/2:
#                         # left side of GD barrier
#                         potential_well_arr[pos] = right_tanh_function(position, GD_barrier_height, GD_barrier_start, Kappa)

#                 elif (GD_barrier_start + GD_barrier_end)/2 <= position <= (GD_barrier_end + drain_well_end)/2:
#                         # right side of GD barrier
#                         potential_well_arr[pos] = left_tanh_function(position, GD_barrier_height, GD_barrier_end, Kappa)

#                 elif (GD_barrier_end + drain_well_end)/2 <= position <= position_end:
#                         # infinite barrier on the right side
#                         potential_well_arr[pos] = right_tanh_function(position, infinite_barrier, drain_well_end, INFINITE_STEP_SMOOTHNESS)

#         return potential_well_arr*10**3*2*PI*H_BAR # Joule

# N = 2**12
# position_start = -30*1.e-6/x_s # micrometer
# position_end   = 60*1.e-6/x_s # micrometer

# complete_transistor_position  = np.linspace(position_start, position_end, N)

# INFINITE_STEP_SMOOTHNESS = 0.0001
# FINITE_STEP_SMOOTHNESS = 0.006 # less will give sharper transition

# # positions in dimensionless form
# source_well_start = -20*1.e-6/x_s 
# SG_barrier_start = -3.5*1.e-6/x_s
# SG_barrier_end =    0*1.e-6/x_s
# GD_barrier_start = 4*1.e-6/x_s
# GD_barrier_end = 7*1.e-6/x_s
# drain_well_end = 50*1.e-6/x_s

# complete_transistor_potential = smooth_potential_well(complete_transistor_position, source_well_start, SG_barrier_start, SG_barrier_height_kHz, SG_barrier_end, GD_barrier_start,GD_barrier_height_kHz, GD_barrier_end, drain_well_end, FINITE_STEP_SMOOTHNESS)
# fig, ax = plt.subplots()
# #fig = plt.figure()
# fig.set_figwidth(16)
# fig.set_figheight(6)
# plt.plot(complete_transistor_position, complete_transistor_potential/(10**3*2*PI*H_BAR),linewidth = 3,
#         label = "Potential landscape", color="tab:red")
# plt.axvline(x=source_well_start, color="k", linestyle='--')
# plt.axvline(x=SG_barrier_start, color="k", linestyle='--')
# plt.axvline(x=SG_barrier_end, color="k", linestyle='--')
# plt.axvline(x=GD_barrier_start, color="k", linestyle='--')
# plt.axvline(x=GD_barrier_end, color="k", linestyle='--')
# plt.ylim([0,GD_barrier_height_kHz*1.02])
# plt.axhline(y = GD_barrier_height_kHz)
# plt.axhline(y = SG_barrier_height_kHz)
# #plt.xlim([-6,12])
# plt.ylabel(r"Potential, $(J)$",labelpad=10)
# plt.xlabel(r"Position, $(\tilde{x})$",labelpad=10)
# fig.tight_layout(pad=1.0)

# path = "/Users/sasankadowarah/atomtronics/"
# os.chdir(path)
# np.save("transistor_position.npy",  complete_transistor_position*x_s)
# np.save("transistor_potential.npy", complete_transistor_potential)

# ax = fig.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(2)
# plt.gcf().subplots_adjust(bottom=0.2)
# plt.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
# plt.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
# plt.legend()
# plt.show()

# %% [markdown]
# #### Transistor potential Gaussian barrier

# %%
PI = np.pi
H_BAR = 6.626*10**(-34)/(2*PI)

source_well_bias_potential_lst = np.around(np.linspace(18,27,16*4),2)#[i for i in range(32)]
np.save("V_SS_lst.npy", source_well_bias_potential_lst)

source_well_bias_potential_index = int(sys.argv[1])
V_SS = source_well_bias_potential_lst[source_well_bias_potential_index]

V_INFINITE_BARRIER  = 3000
np.save("V_inf.npy", V_INFINITE_BARRIER)

N = 2**13
np.save("N.npy",N)

position_start      = -20
position_end        = 60
# positions in SI units
source_well_start   = -15
gate_well_start     = 0
gate_well_end       = 4.8
drain_well_end      = 50

np.save("position_start.npy",position_start)
np.save("position_end.npy",position_end)
np.save("source_well_start.npy",source_well_start)
np.save("gate_well_start.npy",gate_well_start)
np.save("gate_well_end.npy",gate_well_end)

def left_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 - barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

def right_tanh_function(xs, barrier_height, x_0, smoothness_control_parameter):
                return barrier_height/2 + barrier_height/2 * np.tanh((xs-x_0)/(barrier_height*smoothness_control_parameter))

"""
This function takes three points as input and returns the coefficients
of the quadratic function passing through these points.
"""
def harmonic_well(x1,y1,x2,y2,x3,y3):
     A = np.array([[x1**2, x1, 1],
                   [x2**2, x2, 1],
                   [x3**2, x3, 1]])
     b = np.array([y1, y2, y3])
     c1, c2, c3 = np.linalg.solve(A, b)
     return c1, c2, c3

def gaussian_barrier(x, mu, SG_barrier_height, GD_barrier_height, sigma):

     def f(x, mu, barrier_height, sigma):
          return barrier_height*np.exp(-((x-mu)/sigma)**2)

     # gaussian = (single_gaussian_barrier(x, gate_well_start, SG_barrier_height, sigma) +
     #            single_gaussian_barrier(x, gate_well_end, GD_barrier_height, sigma)) 

     gaussian = (SG_barrier_height*np.exp(-((x-gate_well_start)/sigma)**2) +
                GD_barrier_height*np.exp(-((x-gate_well_end)/sigma)**2))

     # making the gate well harmonic
     delta = 0.35
     # f(x) = a*x^2 + b*x + c
     p, q, r = harmonic_well(
               gate_well_start + delta, f(gate_well_start + delta,gate_well_start, SG_barrier_height, sigma),
               gate_well_end - delta, f(gate_well_end - delta, gate_well_end ,
               GD_barrier_height, sigma),
               (gate_well_start+gate_well_end)/2, 0)
 
     gaussian = np.where((x > (gate_well_start + delta)) & (x < (gate_well_end-delta)),
                p*xs**2 + q*xs + r, gaussian)

     # corodinate where the source well intersects the SG barrier
     plataeu_x_cor =  (mu - sigma*np.sqrt(-np.log(V_SS/SG_barrier_height)))

     # bias potential in the source well
     gaussian = np.where(x < plataeu_x_cor, V_INFINITE_BARRIER/2 - V_INFINITE_BARRIER/2 * np.tanh((xs-source_well_start)/(0.0005*V_INFINITE_BARRIER)) + V_SS, gaussian)   

     # infinite barrier at the left end of the source well
     gaussian = np.where(x < -30, V_INFINITE_BARRIER, gaussian)

     # infinite barrier at the end of the drain well
     gaussian = np.where(x > (gate_well_end+drain_well_end)/2, right_tanh_function(xs, V_INFINITE_BARRIER, drain_well_end, 0.0005), gaussian)

     return gaussian

 

fig, ax = plt.subplots()
fig.set_figwidth(20)
fig.set_figheight(6)
xs = np.linspace(position_start,position_end,N)
# changing position to SI units
xs_SI = xs*1.e-6
# changing potential to SI units
complete_transistor_potential = gaussian_barrier(xs,0,30,32,0.9)
complete_transistor_potential_SI = complete_transistor_potential*10**3*2*PI*H_BAR


plt.ylim([0,33*10**3*2*PI*H_BAR*1.02])
plt.ylabel(r"Potential, $V(x)$",labelpad=10)
plt.xlabel(r"Position, $x$",labelpad=10)
fig.tight_layout(pad=1.0)
#path = "/Users/sasankadowarah/atomtronics/"
#os.chdir(path)
np.save("transistor_position_gaussian.npy",  xs_SI)
np.save("transistor_potential_gaussian.npy", complete_transistor_potential_SI)
#plt.show()"""

# %% [markdown]
# #### Source well potential

position_start = xs_SI[0]

# %%
complete_transistor_position = xs_SI/x_s
source_well_position = complete_transistor_position[np.where((complete_transistor_position > position_start/x_s) & (complete_transistor_position < gate_well_start/x_s))]
source_well_potential = complete_transistor_potential_SI[0:len(source_well_position)]

np.save("source_well_position.npy",source_well_position)
np.save("source_well_potential.npy",source_well_potential)


L  = np.abs((source_well_position[-1]-source_well_position[0]))
N = len(source_well_position)
dx = L/N
np.save("dx1.npy",dx)
dk = (2*PI)/L

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

# %% [markdown]
# #### imaginary time evolution for ground state in the source well

# %%
# start with an initial state
psi_initial = np.ones(N)
psi_initial = normalize_x(psi_initial) 

# wavefunction is evolved in imaginary time to get the ground state
final_time_SI = 1.e-1
time_step_SI  = -1j*10**(-7)   
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
psi_source_well_ITE = time_split_suzukui_trotter(psi_initial,source_well_potential,time_step,final_time, [])
np.save("source_well_ground_state.npy",psi_source_well_ITE)
#print("Normalization of the wavefucntion = ",np.sqrt(np.sum(np.abs(psi_source_well_ITE)**2)*dx) )

# %% [markdown]
# #### plot ground state wavefunction  in the source well

# %%
data0 = source_well_position
data1 = np.abs(psi_source_well_ITE)**2*dx
data3 = source_well_potential#/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r"Position, $\tilde{x}$", labelpad=20)
ax1.set_ylabel(r"$|\tilde{\psi}|^{2}$", color="tab:red", labelpad=20)
ax1.plot(data0, data1, color="tab:red",linewidth = 5)
plt.title(r"Ground state wavefunction in the source well")
#plt.legend()
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel(r"$V(x)/(\epsilon \omega^{2}_{x} x^{2}_{s})$ ", color=color,  labelpad=20)
ax2.plot(data0, data3, color=color,linewidth = 5)
ax1.set_ylim([0,0.003])
ax1.set_xlim([-3,0.1])
ax2.set_ylim([21*10**3*2*PI*H_BAR,33*10**3*2*PI*H_BAR])
ax2.tick_params(axis="y", labelcolor=color)
fig.set_figwidth(12)
fig.set_figheight(6)
plt.subplots_adjust(bottom=0.2)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
plt.savefig("ground_state_in_source_well_"+str(NUMBER_OF_ATOMS)+".jpg", dpi=300)
fig.tight_layout()

# %% [markdown]
# #### chemical potential of the BEC

# %%
data0 = source_well_position
data1 = source_well_potential +g_source*NUMBER_OF_ATOMS*np.abs(psi_source_well_ITE/np.sqrt(x_s))**2
data3 = source_well_potential 
fig, ax1 = plt.subplots()

np.save("chemical_potential_in_source_well.npy",data1)


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
ax1.set_xlim([-4,0])
plt.savefig("chemical_potential_in_source_well_"+str(NUMBER_OF_ATOMS)+".jpg", dpi=300)
fig.tight_layout()

# %% [markdown]
# ### real time evolution in the complete transistor potential

# %% [markdown]
# #### time split for real time evolution

# %%
N = len(complete_transistor_position)
L = np.abs((complete_transistor_position[-1]-complete_transistor_position[0]))
dx = L/N
np.save("dx2.npy",dx)
# momentum space discretization
dk = (2*PI)/L
k = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk
    
E_k = k**2*epsilon/2

# put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential = psi_source_well_ITE
while len(psi_initial_for_full_potential) < N:
    psi_initial_for_full_potential = np.hstack((psi_initial_for_full_potential,np.array([0])))
    

final_time_SI = 25*10**(-3)
time_step_SI  = 10**(-8)  
# time is made dimensionless  
final_time = OMEGA_X*final_time_SI
time_step = OMEGA_X*time_step_SI
time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential,
                                        complete_transistor_potential*10**3*2*PI*H_BAR,
                                        time_step, final_time, [t for t in range(0,50,1)])

# %%
# plotting everything in SI units
data0 = complete_transistor_position*x_s*1.e6
data1 = np.abs(time_evolved_wavefunction_time_split)**2*dx
#data2 = np.abs(time_evolved_wavefunction_time_split)**2
data3 = complete_transistor_potential#/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)
fig, ax1 = plt.subplots()
ax1.set_xlabel(r"Position, $x (\mu m)$")
ax1.set_ylabel(r"$|\psi(x)|^{2}$", color="tab:red", labelpad = 20)
ax1.plot(data0, data1, color="tab:red",linewidth = 3.4)
#plt.legend()
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()
color = "tab:blue"
ax2.plot(data0, data3, color=color,linewidth = 7)
#ax2.set_ylim([0, (GD_barrier_height_kHz*10**3*2*PI*H_BAR)/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)*1.02])
ax2.set_ylim([0, 33*1.02])
#ax2.set_ylabel(r"$V(x)/(\epsilon \omega^{2}_{x} x^{2}_{s})$ ", color=color, labelpad = 20)
#ax2.set_ylim([0, (GD_barrier_height_kHz*10**3*2*PI*H_BAR)*1.02])
ax2.set_ylabel(r"Potential, $V(x)$ (J) ", color=color, labelpad = 20)
ax2.tick_params(axis="y", labelcolor=color)
plt.title(r"Time, "+"$ t ="+str(final_time_SI*1.e3)+"(ms), N_{atom} =" + str(NUMBER_OF_ATOMS)+"$")
fig.set_figwidth(23)
fig.set_figheight(9)
plt.subplots_adjust(bottom=0.2)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
ax1.tick_params(axis="x", direction="inout", length=20, width=2, color="k")
ax1.tick_params(axis="y", direction="inout", length=20, width=2, color="k")
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())    
ax1.tick_params(which="minor", length=10, width=1, direction='in')
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax1.axvline(x=source_well_start*x_s*1.e6, color="k", linestyle='--')
ax2.tick_params(which="minor", length=10, width=1, direction='in')
ax1.set_xlim([-2*1.e-5*1.e6, 5*1.e-5*1.e6])
plt.savefig("psi_"+str(NUMBER_OF_ATOMS)+"_"+str(final_time_SI*1.e3)+"(ms).jpg", dpi=600)
fig.tight_layout()

# %%
# print("Total number of atoms in the trap = ", np.sum(NUMBER_OF_ATOMS*np.abs(time_evolved_wavefunction_time_split)**2*dx))

# %%
# # function to calculate the number of atoms in each well
# def number_of_atom(wavefunction,x1,x2):
#     psi_lst = []
#     for i in range(N):
#         if x1 <= complete_transistor_position[i] <= x2:
#             psi_lst.append(wavefunction[i])
#     return NUMBER_OF_ATOMS*np.sum(np.abs(psi_lst)**2)*dx

# # positions in dimensionless form
# source_well_start = -20*1.e-6/x_s 
# SG_barrier_start = -3*1.e-6/x_s
# SG_barrier_end = 0*1.e-6/x_s
# GD_barrier_start = 3*1.e-6/x_s
# GD_barrier_end = 6*1.e-6/x_s
# drain_well_end = 50*1.e-6/x_s
# print("Number of atoms in source well =",
#       number_of_atom(time_evolved_wavefunction_time_split,source_well_start, SG_barrier_end))
# print("Number of atoms in gate well =",
#       number_of_atom(time_evolved_wavefunction_time_split, SG_barrier_start, GD_barrier_end))
# print("Number of atoms in drain well =",
#       number_of_atom(time_evolved_wavefunction_time_split,GD_barrier_end ,drain_well_end))    

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


