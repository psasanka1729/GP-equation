import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import glob
import matplotlib.ticker as ticker

# matplotlib parameters 
large = 40; med = 30; small = 20
params = {'axes.titlesize': med,
          'axes.titlepad' : med,
          'legend.fontsize': med,
          'axes.labelsize': med ,
          'axes.titlesize': med ,
          'xtick.labelsize': med ,
          'ytick.labelsize': med ,
          'figure.titlesize': med}
plt.rcParams.update(params)

b_index = int(sys.argv[1])

path = "/home/sxd190113/scratch/imaginary_time_evolution_wavefunction_data/b" + str(b_index)
os.chdir(path) 

position_arr = np.load("transistor_position_arr.npy") # in meters.
complete_transistor_potential = np.load("transistor_potential_arr.npy")
time_evolved_wavefunction_time_split = np.load("wavefunction_time_evolved_289.9ms.npy")

PI = np.pi
H_BAR = 1.0545718 * 10 ** (-34)
barrier_height_GD = 33
dx_dimless = 0.20420328825905057
def create_wavefunction_animation(x_axis_limit, ylim, label):
    # Parameters
    tmax = 299.0  # Final time in milliseconds
    dt = 1.e-1  # Time interval in milliseconds

    # Load potential data
    global complete_transistor_potential  # Potential data, defined elsewhere

    time_lst = np.arange(0, tmax, dt)
    # Get sorted list of wavefunction files
    wavefunction_files = ["wavefunction_time_evolved_{:.1f}ms.npy".format(t) for t in np.arange(0, tmax, dt)]

    # Set up the figure and the axes
    fig, ax1 = plt.subplots()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    ax1.set_xlabel(r"Position, $x$ (μm)", labelpad=10)
    ax1.set_ylabel(r"Probability density, $|\psi|^{2} dx$", color="tab:red", labelpad=10)
    ax1.set_xlim(x_axis_limit)

    # Secondary axis for potential
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Potential, $V(x)$ (kHz) ", color="tab:blue", labelpad=10)
    ax2.plot(position_arr * 1.e6, complete_transistor_potential / (10**3 * 2 * PI * H_BAR), color="tab:blue", linewidth=2.1)
    ax2.axvline(x=40, color="k", linestyle="--", linewidth=1)
    ax2.set_xlim(x_axis_limit)
    #ax1.set_ylim([0, 1.2*np.abs(np.max(time_evolved_wavefunction_time_split[(position_arr > well_position_1*1.e-6) & (position_arr < well_position_2*1.e-6)] ))**2*dx_dimless])
    ax2.set_ylim([0, barrier_height_GD * 1.2])  # In kHz units.
    ax2.tick_params(axis="y", labelcolor="k")
    ax1.axhline(y=0, color="k", linestyle='--')

    # Initialize wavefunction line object
    line_wavefunction, = ax1.plot([], [], color="tab:red", linewidth=2.5, label="Wavefunction")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Axis styling for minor ticks
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.tick_params(axis="x", direction="inout", length=10, width=2, color="k")
    ax1.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
    ax2.tick_params(axis="y", direction="inout", length=10, width=2, color="k")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.tick_params(which="minor", length=5, width=1, direction="in")
    ax2.tick_params(which="minor", length=5, width=1, direction="in")

    # Initialize animation function
    # Initialize text object for time annotation
    time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=med, verticalalignment='top')

    # Initialize animation function
    def init():
        line_wavefunction.set_data([], [])
        time_text.set_text('')
        return line_wavefunction, time_text
    # Update function for each frame
    def update(frame):
        path = "/home/sxd190113/scratch/imaginary_time_evolution_wavefunction_data/b" + str(b_index)
        os.chdir(path)
        wavefunction = np.load(wavefunction_files[frame])
        line_wavefunction.set_data(position_arr * 1.e6, np.abs(wavefunction)**2 * dx_dimless)  # Adjust scaling 
        #max_wavefunction_value = np.max(np.abs(wavefunction[(position_arr > well_position_1*1.e-6)&(position_arr< well_position_2*1.e-6)])**2 * dx_dimless)
        ax1.set_ylim(ylim)        
        time_text.set_text(f'Time = {time_lst[frame]:.1f} ms')
        return line_wavefunction, time_text

    # Create the animation with a longer display time per frame
    ani = FuncAnimation(fig, update, frames=len(wavefunction_files), init_func=init, blit=True)

    # Save the movie with a reduced frame rate for a slower animation
    #ani.save("wavefunction_evolution.mp4", fps=5, extra_args=['-vcodec', 'libx264'])
    pillow_writer = PillowWriter(fps=15)
    ani.save("wavefunction_evolution_"+str(label)+"_15.gif", writer=pillow_writer)
    #pillow_writer = PillowWriter(fps=30)
    #ani.save("wavefunction_evolution_"+str(label)+"_30.gif", writer=pillow_writer)    
    # Show the animation
    plt.close()

# Example usage
create_wavefunction_animation([4.8, 1000], [0, 2*1.e-7], "drain")  # Adjust limits as needed
create_wavefunction_animation([-40,10], [0,0.005], "source")
