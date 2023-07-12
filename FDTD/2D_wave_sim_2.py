import numpy as np
from math import exp
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d

ie = 100
je = 100
ic = int(ie / 2)
jc = int(je / 2)

ez = np.zeros((ie, je))
dz = np.zeros((ie, je))
hx = np.zeros((ie, je))
hy = np.zeros((ie, je))

ddx = 0.01 # Cell size
dt = ddx / 6e8 # Time step size

# Create Dielectric Profile
epsz = 8.854e-12

# Pulse Parameters
t0 = 20
spread = 6

gaz = np.ones((ie, je))

nsteps = 200

# Dictionary to keep track of desired points for plotting
plotting_points = [
{'label': 'a', 'num_steps': 40, 'data_to_plot': None},
{'label': 'b', 'num_steps': 80, 'data_to_plot': None},
{'label': 'c', 'num_steps': 150, 'data_to_plot': None},
{'label': 'd', 'num_steps': 200, 'data_to_plot': None},
]

# Main FDTD Loop
for time_step in range(1, nsteps + 1):
    # Calculate Dz
    for j in range(1, je):
        for i in range(1, ie):
            dz[i, j] = dz[i, j] + 0.5 * (hy[i, j] - hy[i - 1, j] -
                       hx[i, j] + hx[i, j - 1])

    # Put a Gaussian pulse in the middle
    pulse = exp(-0.5 * ((t0 - time_step) / spread) ** 2)
    dz[ic, jc] = pulse

    # Calculate the Ez field from Dz
    for j in range(1, je):
        for i in range(1, ie):
            ez[i, j] = gaz[i, j] * dz[i, j]

    # Calculate the Hx field
    for j in range(je - 1):
        for i in range(ie - 1):
            hx[i, j] = hx[i, j] + 0.5 * (ez[i, j] - ez[i, j + 1])

    # Calculate the Hy field
    for j in range(je - 1):
        for i in range(ie - 1):
            hy[i, j] = hy[i, j] + 0.5 * (ez[i + 1, j] - ez[i, j])

    # Save data at certain points for later plotting
    for plotting_point in plotting_points:
        if time_step == plotting_point['num_steps']:
            plotting_point['data_to_plot'] = np.copy(ez)


fig = plt.figure(figsize=(8, 7))
X, Y = np.meshgrid(range(je), range(ie))

def plot_e_field(ax, data, timestep, label):
    """2d Plot of E field at a single time step"""
    
    ax.imshow(data[:, :])
    


# Plot the E field at each of the four time steps saved earlier
for subplot_num, plotting_point in enumerate(plotting_points):
    ax = fig.add_subplot(2, 2, subplot_num + 1)
    plot_e_field(ax, plotting_point['data_to_plot'],
    plotting_point['num_steps'],
    plotting_point['label'])

plt.subplots_adjust(bottom=0.05, left=0.10, hspace=0.05)
plt.show()