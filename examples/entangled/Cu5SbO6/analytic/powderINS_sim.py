import numpy as np
import matplotlib.pyplot as plt
import os

# ---- Import Q and E range for simulation ----
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, 'map_200K.txt')

# Load data
data = np.loadtxt(file_name, skiprows=1)
Q_col = data[:, 0]
w_col = data[:, 1]

# ---- Coupling parameters in meV ----
J1 = 16.5
J2 = -6
J3 = 0
J4 = 3
ε = J1/40

# ---- Geometry parameters for CSO ----
l1 = 9 # in Å
l2 = 0.59 * l1
l3 = 0.73 * l1
phi_cso = 30 * (np.pi / 180)
theta_cso = 62 * (np.pi / 180)

# ---- Coordinate transformations ----
theta = np.linspace(0, np.pi, 1000)
phi = np.linspace(0, 2 * np.pi, 1000)

theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
sin_theta = np.sin(theta_grid)
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]

# Coefficients
A = 0.0232
B = 0.1502
C = 0.8305
D = -0.0039

a = 34.488
b = 13.113
c = 5.392

# ----Spherical averaging of S(Q, w) ----
I_all = np.zeros_like(Q_col)
unique_Qs = np.unique(Q_col)

for q_val in unique_Qs:
    mask = (Q_col == q_val)
    w_sub = w_col[mask]

    # Rescale Q for simulation
    Q_sim = q_val * l1

    Q_val = Q_sim
    Qx = Q_val * sin_theta * np.cos(phi_grid)
    Qy = Q_val * sin_theta * np.sin(phi_grid)
    Qz = Q_val * np.cos(theta_grid)

    q1 = (1/(2 * np.pi)) * Qx
    q2 = ((l2/l1)/(2 * np.pi)) * (np.sin(phi_cso) * Qy + np.cos(phi_cso) * Qx)
    q3 = ((l3/l1)/(2 * np.pi)) * (np.sin(theta_cso) * Qz + np.cos(theta_cso) * Qx)

    E = J1 - (J2/2) * np.cos(2 * np.pi * q1) - J3 * np.cos(np.pi * q1) * np.cos(np.pi * (2 * q2 - q1)) - (J4/2) * np.cos(2 * np.pi * (q3 - q1))

    S_pre = (1/np.pi) * (1 - np.cos(4 * np.pi * q1/3))

    # Broadcast w_sub against E
    w_sub_col = w_sub[:, np.newaxis, np.newaxis]
    S_slice = S_pre[np.newaxis, :, :] * (ε/((w_sub_col - E[np.newaxis, :, :])**2 + ε**2))

    # Integrate over theta and phi
    S_avg_sub = np.sum(S_slice * sin_theta[np.newaxis, :, :], axis=(1, 2)) * dtheta * dphi / (4 * np.pi)

    # Form factor
    s = q_val / (4 * np.pi)
    F = A * np.exp(- a * s**2) + B * np.exp(- b * s**2) + C * np.exp(- c * s**2) + D

    # Intensity
    I_sub = (F**2) * S_avg_sub
    I_all[mask] = I_sub

# ---- Export simulated result ----
output_path = os.path.join(script_dir, 'simulated_map.csv')
np.savetxt(output_path, np.column_stack((Q_col, w_col, I_all)), delimiter=',', header='Q,E,I', comments='')
print(f"Exported simulated data to: {output_path}")

# ---- Plotting ----
plt.figure(figsize=(8, 6))
plt.tricontourf(Q_col, w_col, I_all, 50, cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('Q ($\AA^{-1}$)')
plt.ylabel('Energy (meV)')
plt.xlim(0.2, 5.65)
plt.ylim(0, 21)
text_str = f'$J_1={J1}$ meV$, J_2={J2}$ meV$, J_3={J3}$ meV$, J_4={J4}$ meV\n$\epsilon=J1/{J1/ε:.2f}$'
plt.text(0.0, 1.02, text_str, transform=plt.gca().transAxes, verticalalignment='bottom', horizontalalignment='left')
plt.show()
