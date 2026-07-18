import numpy as np
import matplotlib.pyplot as plt
import os


# Exchange parameter list
J1 = 16.5 # in meV
J2 = -0.36 * J1
J4 = 0.18 * J1

# Directions
# H scan with L = fixed
# H = np.linspace(0, 4, 500)
# L = np.linspace(1, 1, 500)

# L scan with H = fixed
H = np.linspace(0.5, 0.5, 500)
L = np.linspace(0, 4, 500)

# Dispersion relation
E = J1 - (J2 / 2) * np.cos(2 * np.pi * H) - (J4 / 2) * np.cos(2 * np.pi * (L - H))

# Structure factor
S = 1 - np.cos((4 * np.pi / 3) * H)

# --- Save data to CSV ---
# Get the directory where the script is located to save the output there.
script_dir = os.path.dirname(os.path.abspath(__file__))

# The data for the H scan (E and S vs L) will be saved to a CSV file.
# output_filename = "dispersion_q1_1p0_scan.csv"
# output_filepath = os.path.join(script_dir, output_filename)
# output_data = np.column_stack((H, E, S))
# np.savetxt(output_filepath, output_data, delimiter=',', header='H,Energy_meV,Strucure_factor_arb_units', comments='')
# print(f"Saved dispersion data to {output_filepath}")

# The data for the L scan (E and S vs L) will be saved to a CSV file.
output_filename = "dispersion_0p5_q3_scan.csv"
output_filepath = os.path.join(script_dir, output_filename)
output_data = np.column_stack((L, E, S))
np.savetxt(output_filepath, output_data, delimiter=',', header='L,Energy_meV,Strucure_factor_arb_units', comments='')
print(f"Saved dispersion data to {output_filepath}")

# Plot results
fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
fig.suptitle('Dispersion Relations', fontsize=16, y=0.92)

axs[0].set_ylim(10, 23)
axs[0].set_xlim(0, 4)
axs[0].set_ylabel('Energy Transfer (meV)')
# axs[0].set_xlabel('(q1, 0, 1)')
# axs[0].plot(H, E, '-')
axs[0].set_xlabel('(0.5, 0, q3)')
axs[0].plot(L, E, '-')

axs[1].set_ylim(0, 2)
axs[1].set_xlim(0, 4)
axs[1].set_ylabel('Structure factor (arb. units.)')
# axs[1].set_xlabel('(q1, 0, 1)')
# axs[1].plot(H, S, '-')
axs[1].set_xlabel('(0.5, 0, q3)')
axs[1].plot(L, S, '-')

plt.show()