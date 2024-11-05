import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import glob
import matplotlib.ticker as ticker

def create_wavefunction_animation(x_axis_limit, y_axis_limit):
    # Parameters
    tmax = 299  # Final time in milliseconds
    dt = 0.1  # Time interval in milliseconds

    # Load potential data
    global complete_transistor_potential  # Potential data, defined elsewhere

    # Get sorted list of wavefunction files
    wavefunction_files = ["wavefunction_time_evolved_{:.3f}ms.npy".format(t) for t in np.arange(0, tmax, dt)]

    # Set up the figure and the axes
    fig, ax1 = plt.subplots()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    ax1.set_xlabel(r"Position, $x$ (μm)", labelpad=10)
    ax1.set_ylabel(r"Wavefunction, $|\psi|^{2}$", color="tab:red", labelpad=10)
    ax1.set_xlim([-40, 30])

    # Secondary axis for potential
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Potential, $V(x)$", color="tab:blue", labelpad=10)
    ax2.plot(position_arr * 1.e6, complete_transistor_potential / (10**3 * 2 * PI * H_BAR), color="tab:blue", linewidth=3.1)
    ax2.set_xlim(x_axis_limit)
    ax1.set_ylim(y_axis_limit)
    ax2.set_ylim([0, barrier_height_GD * 1.2])  # In kHz units.
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.axhline(y=0, color="k", linestyle='--')

    # Initialize wavefunction line object
    line_wavefunction, = ax1.plot([], [], color="tab:red", linewidth=3.2, label="Wavefunction")
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
    def init():
        line_wavefunction.set_data([], [])
        return line_wavefunction,

    # Update function for each frame
    def update(frame):
        path = "/home/sxd190113/scratch/imaginary_time_evolution/b" + str(source_bias_index)
        wavefunction = np.load(wavefunction_files[frame])
        line_wavefunction.set_data(position_arr * 1.e6, np.abs(wavefunction)**2 * solver_complete_potential.dx_dimless)  # Adjust scaling
        return line_wavefunction,

    # Create the animation with a longer display time per frame
    ani = FuncAnimation(fig, update, frames=len(wavefunction_files), init_func=init, blit=True)

    # Save the movie with a reduced frame rate for a slower animation
    ani.save("wavefunction_evolution_10.gif", fps=10, writer=PillowWriter())
    ani.save("wavefunction_evolution_60.gif", fps=60, writer=PillowWriter())
    #ani.save("wavefunction_evolutio_10.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
    #ani.save("wavefunction_evolution_60.mp4", fps=60, extra_args=['-vcodec', 'libx264'])
    plt.close()

# Example usage
create_wavefunction_animation([4.8, 1000], [0, 1.e-6])  # Adjust limits as needed
create_wavefunction_animation([-40,10],[0,0.004])
