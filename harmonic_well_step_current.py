# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import fftpack
#from scipy.sparse import csr_matrix
#import scipy.sparse

# %%
# matplotlib parameters 
large = 40; med = 20; small = 20
params = {'axes.titlesize': med,
          'axes.titlepad' : med,
          'legend.fontsize': med,
          'axes.labelsize': med ,
          'axes.titlesize': med ,
          'xtick.labelsize': med ,
          'ytick.labelsize': med ,
          'figure.titlesize': med}
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.serif"] = ["Helvetica Neue"]          
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

# Transistor trap parameters.
TRAP_FREQUENCY = 2*PI*70 #918 # Hz
OMEGA_X = TRAP_FREQUENCY
TRAP_LENGTH = np.sqrt(H_BAR/(ATOM_MASS*TRAP_FREQUENCY)) # m
CROSS_SECTIONAL_AREA = PI*TRAP_LENGTH**2 # m*m

NUMBER_OF_ATOMS = 50000

# Interaction strength in the source well.
g_source   = (4*PI*H_BAR**2*a_s)/(CROSS_SECTIONAL_AREA*ATOM_MASS)
print("g_source = ",g_source)
a_0 = np.sqrt(H_BAR/(TRAP_FREQUENCY*ATOM_MASS))
print(4*PI*a_s*NUMBER_OF_ATOMS, "must be greater than ", np.sqrt(H_BAR/(TRAP_FREQUENCY*ATOM_MASS)))
x_s = (4*np.pi*a_s*NUMBER_OF_ATOMS*a_0**4)**(1/5)    
path = "/Users/sasankadowarah/atomtronics/cluster-codes/harmonic_gate_well/jupyter_notebook_data/"
os.chdir(path)
np.save("x_s.npy",x_s)
#print("x_s = ",x_s)
epsilon = (H_BAR/(ATOM_MASS*OMEGA_X*x_s**2))
#print("epsilon = ",epsilon)
delta   = (g_source*NUMBER_OF_ATOMS*(x_s**2))/(a_0**3*H_BAR*OMEGA_X)
#print("delta =", delta)

# %% [markdown]
# #### Transistor potential Gaussian barrier and harmonic gate well

# %%
N = 2**14

#V_SS = 0.0 # In kHz units.
V_INFINITE_BARRIER  = 1.e4 # In kHz units.

# Position parameters in micrometers.
position_start      = -60
position_end        = 200
source_well_start   = -50
gate_well_start     = 0
gate_well_end       = 4.8#5.224731361650667
drain_well_end      = 190

np.save("position_start.npy",position_start)
np.save("position_end.npy",position_end)
np.save("source_well_start.npy",source_well_start)
np.save("gate_well_start.npy",gate_well_start)
np.save("gate_well_end.npy",gate_well_end)
np.save("drain_well_end.npy",drain_well_end)

bias_potential_in_gate = 0.0 # In kHz units.

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
     return (barrier_height)*np.exp(-((x-x_0)/sigma_1)**2) + (left_plataeu_height + right_plataeu_height)/2 - (left_plataeu_height - right_plataeu_height)/2 * np.tanh((x-x_0)/(sigma_2))

""" Creating the source well.
"""
def source_well_potential_function(x, A, B, C, bias_potential_in_source_well):
     return A*x**2+C*np.exp(-x**2/B)+bias_potential_in_source_well

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

"""
This function creates a potential landscape for a transistor with a source well, gate well, and a drain well.
Input parameters: position_arr - an array of positions in micrometers,
                  SG_barrier_height - barrier height in the source well in kHz units,
                  GD_barrier_height - barrier height in the drain well in kHz units,
                    sigma - smoothness control parameter for the source well barrier,
                    SIGMA_1 - smoothness control parameter for the source well barrier,
                    SIGMA_2 - smoothness control parameter for the source well barrier,
                    SIGMA_3 - smoothness control parameter for the drain well barrier,
                    SIGMA_4 - smoothness control parameter for the drain well barrier.
                    SIGMA_5 - smoothness control parameter for the left side of bias potential in the source well.
"""
def transistor_potential_landscape(V_SS,  position_arr, SG_barrier_height, GD_barrier_height, gate_bias_potential,
     # These parameters control the width of the barriers and the smoothness of the transitions.
     SIGMA_1 = 0.6,
     SIGMA_2 = 0.8,
     SIGMA_3 = 0.6,
     SIGMA_4 = 0.6,

     ):

     delta_left = 0.05
     delta_right = 0.05

     # Creating the source well.
     A = 0.02 # Increasing A results in increase in left side of the source well.
     B = 0.3 # Increasing B results in increase in width of the source well.
     potential = np.zeros(len(position_arr))
     potential = np.where(position_arr <= gate_well_start + delta_left, source_well_potential_function(position_arr, A,B, SG_barrier_height - V_SS,V_SS), potential)

     # First point on the left side of the gate well.
     x_1 = gate_well_start + delta_left
     y_1 = source_well_potential_function(x_1, A, B, SG_barrier_height - V_SS, V_SS)

     # Second point at the end of the gate well.
     x_2 = gate_well_end - delta_right
     y_2 = different_plataeu_gaussian_barrier(gate_well_end - delta_right, gate_well_end, GD_barrier_height, 0,0, SIGMA_3, SIGMA_4)

     # Third point at the center of the gate well.
     x_3 = (gate_well_start + gate_well_end)/2
     y_3 = gate_bias_potential

     """ Harmonic gate well. """
     pp, qq, rr = harmonic_well(x_1, y_1, x_2, y_2, x_3, y_3)

     def harmonic_gate_well(x, pp, qq, rr):
          return pp*x**2 + qq*x + rr

     # This loop creates the harmonic well in the gate well region and the GD barrier.
     for i in range(len(position_arr)):
          if position_arr[i] > gate_well_start + delta_left and position_arr[i] <= gate_well_end - delta_right:
               potential[i] = harmonic_gate_well(position_arr[i], pp, qq, rr)
               #potential[i] = anharmonic_gate_well(position_arr[i], pp, qq, rr, ss, tt)
          elif position_arr[i] >= gate_well_end - delta_right:
               potential[i] = different_plataeu_gaussian_barrier(position_arr, gate_well_end, GD_barrier_height, 0,0, SIGMA_3, SIGMA_4)[i]

     # Creates a barrier at the end of the drain well.
     potential = np.where(position_arr > (gate_well_end+drain_well_end)/2, right_tanh_function(position_arr, V_INFINITE_BARRIER, drain_well_end, 0.5), potential)

     return potential # In kHz units.

# Position array in SI units (meters).
position_arr = np.linspace(position_start,position_end,N)*1.e-6
np.save("transistor_position_arr.npy", position_arr)

barrier_height_SG = 31 # In kHz units.
barrier_height_GD = 32 # In kHz units.

np.save("barrier_height_SG.npy", barrier_height_SG)
np.save("barrier_height_GD.npy", barrier_height_GD)

source_bias = 22.0
complete_transistor_potential = transistor_potential_landscape(source_bias, position_arr*1.e6, barrier_height_SG, barrier_height_GD, bias_potential_in_gate)*10**3*H_BAR*2*PI # In SI units.

np.save("complete_transistor_potential.npy",  complete_transistor_potential)

fig, ax = plt.subplots()
fig.set_figwidth(14)
fig.set_figheight(6)
plt.plot(position_arr*1.e6, complete_transistor_potential/(10**3*H_BAR*2*PI), linewidth = 4, color = "tab:red")
#plt.scatter(position_arr*1.e6, complete_transistor_potential/(10**3*H_BAR), color = "tab:red", s = 10)
#plt.axhline(y=30 , color="k", linestyle='--')   
plt.axhline(y=barrier_height_GD , color="k", linestyle='--', label = r"$V_{GD}$")   
plt.axhline(y=0, color="k", linestyle='--')
# plt.axvline(x=source_well_start, color="k", linestyle='--')
# plt.axvline(x=gate_well_start, color="k", linestyle='--')
# plt.axvline(x=gate_well_end, color="k", linestyle='--')
plt.xlim([-30, 40])
#plt.ylim([0,50*10**(3)*H_BAR*2*PI]) # In SI units.
plt.ylim([0, barrier_height_GD*1.2]) # In kHz units.
plt.ylabel(r"Potential, $V(x),\; (kHz)$",labelpad=10)
plt.xlabel(r"Position, $x, \; (\mu m)$",labelpad=10)
fig.tight_layout(pad=1.0)
# path = "/Users/sasankadowarah/atomtronics/"
# os.chdir(path)
# np.save("transistor_position_gaussian.npy",  xs_SI)
# np.save("transistor_potential_gaussian.npy", complete_transistor_potential_SI)
plt.savefig("complete_transistor_potential_harmonic_gate_well.png", dpi=600)
plt.close()

# %%
# Average density of the gas.
n = NUMBER_OF_ATOMS/(CROSS_SECTIONAL_AREA*(position_end-position_start)*10**(-6))
#print("Gross-Pitaevskii equation is valid if:", n*a_s**3 ,"<< 1") # Ref: Rev. Mod. Phys. 71, 463 (1999)

# %% [markdown]
# #### Source well potential

# %%
""" Like in the experiment; the simulation starts with a very high barrier on the right. 
The GP equation is solved for its ground state in this potential well. Then the right
barrier is lowered to obtain the SG barrier.
 """

position_arr_temp = np.linspace(position_start,position_end,N)
source_well_position_arr = position_arr_temp[0:np.where(np.abs(position_arr_temp- gate_well_start)<1.e-1)[0][0]+1]
source_well_potential = np.zeros(len(source_well_position_arr))
delta_left = 0.05
delta_right = 0.05

# Creating the source well.
A = 0.02 # Increasing A results in increase in left side of the source well.
B = 0.3 # Increasing B results in increase in width of the source well.
initial_SG_barrier_height = 100
V_SS = source_bias
source_well_potential = np.where(source_well_position_arr <= gate_well_start + delta_left, source_well_potential_function(source_well_position_arr, A,B, initial_SG_barrier_height - V_SS,V_SS), source_well_potential)
# Making the position array dimensionless.
transistor_position_arr_dimless = position_arr/x_s
path = "/Users/sasankadowarah/atomtronics/cluster-codes/harmonic_gate_well/jupyter_notebook_data/"
os.chdir(path)
np.save("transistor_position_dimless.npy", transistor_position_arr_dimless)

""" Extracting the source well position array from the original transistor potential."""
# index_of_source_position_start = np.where(np.abs(transistor_position_arr_dimless*x_s - (position_start*1.e-6))< 1.e-7)[0][0]
# index_of_source_position_end = np.where(np.abs(transistor_position_arr_dimless*x_s - (0.02*1.e-6))< 1.e-7)[0][0]
# source_well_position_arr_dimless = transistor_position_arr_dimless[index_of_source_position_start:index_of_source_position_end]
# source_well_potential = complete_transistor_potential[index_of_source_position_start:index_of_source_position_end]

source_well_position_arr_dimless = (source_well_position_arr*1.e-6)/x_s
source_well_potential = source_well_potential*10**3*H_BAR*2*PI

np.save("source_well_position_dimless.npy", source_well_position_arr_dimless)
np.save("source_well_potential.npy", source_well_potential)

f = plt.figure()    
plt.plot(source_well_position_arr_dimless, source_well_potential,linewidth=5)
#plt.ylim([0,barrier_height_GD*10**3*H_BAR*1.02])
ax = f.gca()
#ax.axhline(0, color="green",linestyle="--",linewidth = 2)
plt.xlabel(r"Dimentionless position, $\tilde{x}$", labelpad = 20)
plt.ylabel(r"Potential, $V_{\mathrm{Source}}\; (J)$", labelpad = 20)
f.set_figwidth(10)
f.set_figheight(6)
plt.savefig("source_well.jpg", dpi = 600)
plt.close()

# %%
# f = plt.figure()    
# plt.plot(source_well_position_arr_dimless, source_well_potential,linewidth=3)
# #plt.ylim([0,barrier_height_GD*10**3*H_BAR*1.02])
# ax = f.gca()
# #ax.axhline(0, color="green",linestyle="--",linewidth = 2)
# plt.xlabel(r"Dimentionless position, $\tilde{x}$", labelpad = 20)
# plt.ylabel(r"Potential, $V_{\mathrm{Source}}\; (J)$", labelpad = 20)
# f.set_figwidth(10)
# f.set_figheight(6)
# plt.show()

# %% [markdown]
# #### Single particle energy levels in the gate well

# %%
""" In this section we solve for the single particle energy levels of the harmonic gate well potential.
This is necessary to test the coherence of the emitted matter wave in the drain well. """
transistor_position_arr_dimless = position_arr/x_s
transistor_gate_well_position_dimless = transistor_position_arr_dimless[np.where((transistor_position_arr_dimless >= (gate_well_start*1.e-6/x_s)) & (transistor_position_arr_dimless <= (gate_well_end*1.e-6/x_s)))]
# The perturbation to the Hamiltonian.
OMEGA_HARMONIC = np.sqrt(2*barrier_height_GD*10**3*H_BAR*2*PI/(ATOM_MASS*((transistor_gate_well_position_dimless*x_s)[-1])**2))
# Calculates the length of the source well.
L_dimless  = np.ptp(transistor_gate_well_position_dimless)
# Number of points in the source well.
N = len(transistor_gate_well_position_dimless)
# Discretizing the position space.
dx_dimless = L_dimless/N
np.save("dx_dimless.npy",dx_dimless)
#print("dx (SI) = ", dx_dimless*x_s)
transistor_gate_well_potential = complete_transistor_potential[len(source_well_position_arr_dimless):len(source_well_position_arr_dimless)+len(transistor_gate_well_position_dimless)]

# Shifthing the minimum of the potential to zero.
# index_of_V_minimum = np.argmin(transistor_gate_well_potential)
# transistor_gate_well_potential = transistor_gate_well_potential - transistor_gate_well_potential[index_of_V_minimum]
# transistor_gate_well_position_dimless = transistor_gate_well_position_dimless-transistor_gate_well_position_dimless[index_of_V_minimum]
# plt.plot(transistor_gate_well_position_dimless*x_s, transistor_gate_well_potential)
# OMEGA_HARMONIC = np.sqrt(2*barrier_height_GD*10**3*H_BAR/(ATOM_MASS*((transistor_gate_well_position_dimless*x_s)[-1])**2))
# print("Angular frequency in the gate well  = ", OMEGA_HARMONIC, "Hz")
# plt.plot(transistor_gate_well_position_dimless*x_s,(1/2)*ATOM_MASS*OMEGA_HARMONIC**2*(transistor_gate_well_position_dimless*x_s)**2)
# plt.show()

# %%
# """ Perturbationt theory for the gate well. Calculations are done in SI units. """

# from math import sqrt
# from scipy import special

# def perturbation(anharmonic_potential):
#     return anharmonic_potential-(1/2)*ATOM_MASS*OMEGA_HARMONIC**2*(transistor_gate_well_position_dimless*x_s)**2
#     #return 1.e-13*transistor_gate_well_position_dimless*x_s

# def factorial(n):
#     if n < 0:
#         raise ValueError("Factorial is not defined for negative numbers.")
#     elif n == 0:
#         return 1
#     else:
#         result = 1
#         for i in range(1, n+1):
#             result *= i
#         return result

# # Generate the values of the Hermite polynomial for order n and array x.

# def hermite_polynomial(n, x, monic=False):
#      return special.hermite(n)(x)  
    
# # Returns the wavefunction of the quantum harmonic oscillator.    
# def quantum_harmonic_oscillator_wavefunction(n):
#     x = transistor_gate_well_position_dimless*x_s
#     return (1/(sqrt((2**n)*factorial(n))))*(ATOM_MASS*OMEGA_HARMONIC/(PI*H_BAR))**(1/4)*np.exp(-(ATOM_MASS*OMEGA_HARMONIC/(2*H_BAR))*x**2)*hermite_polynomial(n,sqrt(ATOM_MASS*OMEGA_HARMONIC/H_BAR)*x)

# # Returns the first order energy correction.
# def energy_correction_first_order(n):
#     psi_n = quantum_harmonic_oscillator_wavefunction(n) 
#     return np.dot(psi_n,perturbation(transistor_gate_well_potential)*psi_n)*(dx_dimless*x_s)

# # Energy of n^th energy level of the quantum harmonic oscillator.
# def E(n):
#     return (n+1/2)*H_BAR*OMEGA_HARMONIC

# # Numerator of the sum in the correction.
# def V_mn(m,n):
#     psi_m = quantum_harmonic_oscillator_wavefunction(m)
#     psi_n = quantum_harmonic_oscillator_wavefunction(n)
#     return np.dot(psi_m,perturbation(transistor_gate_well_potential)*psi_n)*(dx_dimless*x_s)

# # Returns the second order correction to the energy.
# def energy_correction_second_order(n):
#     correction = 0.0
#     for m in range(100):
#         if m == n:
#             pass
#         else:
#             correction += np.abs(V_mn(m,n))**2/(E(n)-E(m))
#     return correction    

# def gate_well_energy_from_perturbation_theory(n,epsilon):
#     return (n+1/2)*H_BAR*OMEGA_HARMONIC+epsilon*energy_correction_first_order(n)+epsilon**2*energy_correction_second_order(n)    

# %%
#gate_well_energy_from_perturbation_theory(10,epsilon)/(H_BAR) #/(10**3*H_BAR)

# %%
# Single particle energy levels in gate well.
single_particle_omega = np.sqrt(8*barrier_height_GD*10**3*H_BAR*2*PI/(ATOM_MASS*(gate_well_end*1.e-6 - gate_well_start*1.e-6)**2))
#print(r"Single particle energy level frequency = ", single_particle_omega, "(Hz)")
# Number of single particle energy levels in the gate well.
n_levels = int((barrier_height_GD*10**3*H_BAR*2*PI)/(H_BAR*single_particle_omega) - 1/2)
#print("Number of single particle energy levels in the gate well = ", n_levels)

# %% [markdown]
# #### Time split code to solve the Gross-Pitaevskii equation

# %%
# Calculates the length of the source well.
L_dimless  = np.ptp(source_well_position_arr_dimless)
# Number of points in the source well.
N = len(source_well_position_arr_dimless)
# Discretizing the position space.
dx_dimless = L_dimless/N

#print("dx (SI) = ", dx_dimless*x_s)
# Discretizing the momentum space. dk is dimensionless since the position (L) is dimensionless.
dk_dimless = (2*PI)/L_dimless
#print("dk = ", dk_dimless)
# Total Hamiltonian H = H(k) + H(x) = momentum space part + real space part
""" 
Hamiltonian in position space. 
Inputs: potential_array - potential array in the source well in SI units, 
        psi - wavefunction in the source well in dimensionless units.
Output: Hamiltonian in position space in dimensionless units.
"""
def Hamiltonian_x_dimless(potential_array,psi_dimless): # H(x)
    return potential_array/(epsilon*ATOM_MASS*OMEGA_X**2*x_s**2)+delta*epsilon**(3/2)*np.abs(psi_dimless)**2

""" 
In this section, we will calculate the kinetic energy operator momentum space.
"""
# Momentum space discretization.
k_dimless = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk_dimless
if len(k_dimless) != N:
    k_dimless = np.hstack([np.arange(0,N/2), np.arange(-N/2+1,0)])*dk_dimless
    
# Kinetic energy operator in momentum space.    
E_k_dimless = k_dimless**2*epsilon/2

# Normalize the wavefunction in real space.
def normalize_x(wavefunction_x_dimless):
    return wavefunction_x_dimless/(np.sqrt(np.sum(np.abs(wavefunction_x_dimless)**2)*dx_dimless))

# Normalize the wavefunction in momentum space.
def normalize_k(wavefunction_k_dimless):
    return wavefunction_k_dimless/(np.sqrt(np.sum(np.abs(wavefunction_k_dimless)**2)*dk_dimless))

"""
Time split method for the Suzuki-Trotter decomposition for the time evolution of the wavefunction.
Inputs: initial_wavefunction_dimless - initial wavefunction in the source well in dimensionless units,
        potential_array - potential array in the source well in SI units,
        dt_dimless - time step in dimensionless units,
        total_time_dimless - total time in dimensionless units,
        snapshots_lst - list of times at which the wavefunction is saved as npy file.
Output: time evolved wavefunction in dimensionless units.
"""
def time_split_suzukui_trotter(initial_wavefunction_dimless, potential_array, dt_dimless, total_time_dimless, snapshots_lst):    
    """ 
    Split time Fourier method for the time evolution of the wavefunction.

    \psi(t+dt) = e^(-iH_x*dt/2) e^(-iH_k*dt) e^(-iH_x*dt/2) \psi(t)

    """
    #psi_k = fftpack.fft(initial_wavefunction)
    psi_x_dimless = initial_wavefunction_dimless
    
    total_iterations = int(np.abs(total_time_dimless)/np.abs(dt_dimless))
    print("Number of iterations =", total_iterations)
    
    if snapshots_lst:
        current_time_index = 0
        next_snapshot_time = snapshots_lst[current_time_index]   
     
    for iteration in range(total_iterations):

        time = (dt_dimless*iteration/OMEGA_X)*1.e3 # Changing to SI units miliseconds for convenience.

        # VTV decompostion of Hamiltonian.
        # Evolution in x space.
        psi_x_dimless = np.exp(-Hamiltonian_x_dimless(potential_array,psi_x_dimless) * 1j*dt_dimless/2) * psi_x_dimless

        # Evolution in k space.
        psi_k_dimless = fftpack.fft(psi_x_dimless)        
        psi_k_dimless = np.exp(-(E_k_dimless * 1j*dt_dimless)) * psi_k_dimless

        # Evolution in x space.
        psi_x_dimless = fftpack.ifft(psi_k_dimless) 
        psi_x_dimless = normalize_x(psi_x_dimless)        
        psi_x_dimless = np.exp(-Hamiltonian_x_dimless(potential_array,psi_x_dimless) * 1j*dt_dimless/2) * psi_x_dimless      

        """ This section saves snapshots of the time evolved wavefunction according to the specified times
        in the time_lst. """

        if snapshots_lst:
            # Check if the current time matches the next snapshot time
            if time >= next_snapshot_time:
                np.save("time_evolved_wavefunction_"+str(np.around(time,2))+".npy",psi_x_dimless)
                current_time_index += 1
                if current_time_index < len(time_lst):
                    next_snapshot_time = time_lst[current_time_index]                 

    psi_x_dimless = normalize_x(psi_x_dimless) # Returns the normalized wavefunction.
    
    return psi_x_dimless

# %% [markdown]
# #### imaginary time evolution for ground state in the source well

# %%

# 

# %%
# psi_initial_dimless = np.exp(-(source_well_position_arr_dimless+1.5)**2/(0.1))*np.sqrt(x_s)+np.exp(-(source_well_position_arr_dimless+7.5)**2/(0.1))*np.sqrt(x_s)
# psi_initial_dimless = normalize_x(psi_initial_dimless)
# plt.plot(source_well_position_arr_dimless, np.abs(psi_initial_dimless)**2)
# plt.show()

# %%
# start with an initial state
# psi_initial = np.ones(N)
# psi_initial = normalize_x(psi_initial) 

# psi_initial_dimless = np.exp(-(source_well_position_arr_dimless+1.5)**2/(0.1))*np.sqrt(x_s)+np.exp(-(source_well_position_arr_dimless+2.5)**2/(0.1))*np.sqrt(x_s) #np.ones(N)
# psi_initial_dimless = normalize_x(psi_initial_dimless)

# Wavefunction is evolved in imaginary time to get the ground state.
final_time_SI = 1.e-1 # In seconds unit.
time_step_SI  = -1j*10**(-6) # In seconds unit.
final_time_dimless = OMEGA_X*final_time_SI # Dimensionless time units.
time_step_dimless = OMEGA_X*time_step_SI # Dimensionless time units.

psi_source_well_ITE_dimless = time_split_suzukui_trotter(psi_initial_dimless, source_well_potential, time_step_dimless,final_time_dimless, [])
#print(r"Normalization of the wavefunction $|\psi |^{2}$= ",np.sqrt(np.sum(np.abs(psi_source_well_ITE_dimless)**2)*dx_dimless) )

# %% [markdown]
# #### plot ground state wavefunction  in the source well

# %%
data0 = source_well_position_arr_dimless*x_s
data1 = np.abs(psi_source_well_ITE_dimless)**2*dx_dimless
data3 = source_well_potential/(10**3*H_BAR*2*PI)

np.save("source_well_position_arr_dimless.npy", data0)
np.save("psi_source_well_ITE_dimless.npy", data1)
np.save("source_well_potential.npy", data3)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r"Position, $x(m)$", labelpad=20)
ax1.set_ylabel(r"Wavefunction, $|\psi|^{2}$", color="tab:red", labelpad=10)
ax1.plot(data0, data1, color="tab:red",linewidth = 5)
plt.title(r"Ground state wavefunction in the source well")
#plt.legend()
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel(r"Potential, $\tilde{V}$ ", color=color,  labelpad=20)
ax2.plot(data0, data3, color=color,linewidth = 5)
#ax1.set_xlim([-5,0.1])
#ax2.set_ylim([12,initial_SG_barrier_height*1.2]) 
ax2.tick_params(axis="y", labelcolor=color)
ax1.axhline(y=0, color="k", linestyle='--')
fig.set_figwidth(12)
fig.set_figheight(6)
plt.subplots_adjust(bottom=0.2)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
plt.savefig("ground_state_in_source_well_"+str(NUMBER_OF_ATOMS)+".jpg", dpi=600)
fig.tight_layout()

# %%
ground_state_in_source_well_k_space = fftpack.fft(np.abs(psi_source_well_ITE_dimless)**2)
#ground_state_in_source_well_k_space = fftpack.fftshift(ground_state_in_source_well_k_space)
initial_state_in_k_space_dimless = fftpack.fft(psi_initial_dimless)
#initial_state_in_k_space_dimless = fftpack.fftshift(initial_state_in_k_space_dimless)
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(6)
plt.plot(k_dimless,np.abs(ground_state_in_source_well_k_space)**2*dk_dimless, linewidth = 2, label = "$\psi(k)$")
plt.xlabel(r"$\tilde{k}$")
plt.ylabel(r"$|\tilde{\psi}(\tilde{k})|^{2}$")
plt.plot(k_dimless,np.abs(initial_state_in_k_space_dimless)**2*dk_dimless, linewidth = 2,label = "Initial state")
plt.legend()
plt.title("Source well wavefunction in k space ")
plt.savefig("ground_state_in_the_source_well_in_momentum_space_19.jpg", dpi=600)
plt.close()

# %% [markdown]
# #### chemical potential of the BEC

# %%
psi = psi_source_well_ITE_dimless/np.sqrt(x_s)
kinetic_energy_arr = -(H_BAR**2/(2*ATOM_MASS))*(psi[2:] - 2*psi[1:-1]+psi[:-2])/(dx_dimless*np.sqrt(x_s))**2
#source_well_potential = complete_transistor_potential[index_of_source_position_start:index_of_source_position_end]
potential_energy_arr = source_well_potential*psi+g_source*NUMBER_OF_ATOMS*(np.abs(psi)**2)*psi
#print("Thomas-Fermi approximation is valid if: K.E. = ", np.abs(np.mean(kinetic_energy_arr)) ,"<< P.E. =", np.abs(np.mean(potential_energy_arr))) # Pethick & Smith
#print("Thomas-Fermi approximation is valid if:", NUMBER_OF_ATOMS*a_s/a_0 ,">> 1") # Ref: arxiv: 2003:03590

# %%
data0 = source_well_position_arr_dimless
source_well_potential = complete_transistor_potential[0:len(source_well_position_arr_dimless)]
data1 = source_well_potential +g_source*NUMBER_OF_ATOMS*np.abs(psi_source_well_ITE_dimless/np.sqrt(x_s))**2
data3 = source_well_potential 

fig, ax1 = plt.subplots()

ax1.set_xlabel(r"Position, $\tilde{x}$", labelpad=20)
ax1.set_ylabel(r"Chemical potential $\mu\; (Joules)$", color="tab:red", labelpad=20)
ax1.plot(data0, data1, color="tab:red",linewidth = 5)
plt.title(r"Chemical potential in the source well")
#plt.legend()
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel(r"$V(x)\; (Joules)$ ", color=color,  labelpad=20)
ax2.plot(data0, data3, color=color,linewidth = 5)
ax2.tick_params(axis="y", labelcolor=color)
fig.set_figwidth(10)
fig.set_figheight(6)
plt.subplots_adjust(bottom=0.2)
for spine in ax1.spines.values():
    spine.set_linewidth(2)
ax1.set_xlim([-5,0])    
ax2.set_ylim([0, barrier_height_SG * 10**3*H_BAR*2*PI*1.2 ])
ax1.set_ylim([0, barrier_height_SG*10**3*H_BAR*2*PI*1.2 ])
ax1.axhline(y = barrier_height_GD * 10**3*H_BAR*2*PI , color="k", linestyle='--')
ax1.axhline(y = barrier_height_SG * 10**3*H_BAR*2*PI , color="k", linestyle='--')
ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
#print("Chemical potential in the source well = ", data1[len(data1)//2],"(J) or",data1[int(len(data1)/1.1)]/(H_BAR*10**3*2*PI), "(kHz)")
#ax1.set_xlim([-4,0])
plt.savefig("chemical_potential_in_source_well.jpg", dpi=600)
fig.tight_layout()

# %%
""" This code prints our the end point of the gate well such that the chemical potential
in the source well is equal to the energy of the last single particle energy level in the gate well."""
data1 = source_well_potential +g_source*NUMBER_OF_ATOMS*np.abs(psi_source_well_ITE_dimless/np.sqrt(x_s))**2
mu_s = data1[len(data1)//2]
x_GD = gate_well_start + np.sqrt(8*(barrier_height_SG*10**3*H_BAR*2*PI)/ATOM_MASS)*(H_BAR*(29+1/2)/mu_s)
#print("End point of the gate well = ",x_GD,"(m)")

# %% [markdown]
# #### time split for real time evolution

# %%
N = len(transistor_position_arr_dimless)
L_dimless = np.abs((transistor_position_arr_dimless[-1]-transistor_position_arr_dimless[0]))
dx_dimless = L_dimless/N
np.save("dx_dimless.npy",dx_dimless)
# We do not have to non-dimensionalize dk since transistor position and hence L is already non-dimensional.
dk_dimless = (2*PI)/L_dimless
np.save("dk_dimless.npy",dk_dimless)
#print("dx = ", dx_dimless*x_s)
#print("dk = ", dk_dimless)
# Momentum space discretization done for periodic boundary conditions.
k_dimless = np.hstack([np.arange(0,N/2), np.arange(-N/2,0)])*dk_dimless
# Kinetic energy operator.
E_k_dimless = k_dimless**2*epsilon/2

# Put the initial ground state in the source well of the transistor.
psi_initial_for_full_potential_dimless = psi_source_well_ITE_dimless
while len(psi_initial_for_full_potential_dimless) < N:
    psi_initial_for_full_potential_dimless = np.hstack((psi_initial_for_full_potential_dimless, np.array([0])))

""" We will put a gaussian wave packet in the gate well instead of zero."""
#center_of_gate_well_dimless = ((gate_well_start + gate_well_end)*1.e-6/2)/x_s
#psi_initial_gate_well_dimless = 0.8*1.e2*np.exp(-(transistor_position_arr_dimless - center_of_gate_well_dimless)**2/(0.1))*np.sqrt(x_s)  

#NUMBER_OF_ATOMS = 20000
#psi_initial_for_full_potential_dimless = psi_initial_for_full_potential_dimless+psi_initial_gate_well_dimless
#print("Initial number of atoms in the gate well = ", np.sum(NUMBER_OF_ATOMS*np.abs(psi_initial_gate_well_dimless)**2)*dx_dimless)

# Final time for the wavefunction evolution in units of seconds.
final_time_SI = 100*10**(-3) # In seconds unit.
# Time step to be used in the Suzuki-Trotter decomposition in units of seconds.
time_step_SI  = 10**(-7)  # In seconds unit.

# Time is made dimensionless to be used in the Suzuki-Trotter algorithm.  
final_time_dimless = OMEGA_X*final_time_SI
time_step_dimless = OMEGA_X*time_step_SI

# List of time to save the snapshots of the wavefunction in miliseconds unit.
time_lst = list(np.arange(0.0,int(final_time_SI*1.e3),0.01))
#time_lst = []

time_evolved_wavefunction_time_split = time_split_suzukui_trotter(psi_initial_for_full_potential_dimless,
                                        complete_transistor_potential,
                                        time_step_dimless, final_time_dimless, time_lst)