import numpy as np
from scipy.optimize import minimize
import sympy as sp

# Load atomic positions from spin_model.py
from spin_model import atom_pos, spin_interactions, Hamiltonian, atom_pos_ouc

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to Cartesian coordinates."""
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])

def energy_function(angles, p):
    """Compute the Hamiltonian energy given spin angles."""
    apos_ouc = atom_pos_ouc()  # Get all atom positions including outside unit cell
    nspin_ouc = len(apos_ouc)  # Total number of spins used in Hamiltonian

    # Ensure the length of angles matches the required number of spins
    assert len(angles) == 2 * nspin_ouc, f"Expected {2*nspin_ouc} angles, got {len(angles)}"

    # Convert angles to spin vectors
    Sxyz = [spherical_to_cartesian(angles[2*i], angles[2*i+1]) for i in range(nspin_ouc)]

    # Ensure Sxyz has the correct shape
    assert len(Sxyz) == nspin_ouc, f"Expected {nspin_ouc} spin vectors, got {len(Sxyz)}"

    # Compute Hamiltonian energy
    H_sym = Hamiltonian(Sxyz, p)

    # Convert symbolic expression to numerical function
    H_func = sp.lambdify([], H_sym, "numpy")

    return H_func()  # Evaluate the function


def find_ground_state(p):
    """Find the ground state by minimizing the Hamiltonian."""
    apos_ouc = atom_pos_ouc()
    nspin_ouc = len(apos_ouc)  # Total number of spins

    # Initial angles (random guess)
    initial_angles = np.random.uniform(0, np.pi, 2 * nspin_ouc)

    # Minimize energy
    result = minimize(energy_function, initial_angles, args=(p,), method='L-BFGS-B')

    # Extract optimized angles
    optimized_angles = result.x

    # Convert to Cartesian spins
    optimized_spins = [spherical_to_cartesian(optimized_angles[2*i], optimized_angles[2*i+1]) for i in range(nspin_ouc)]

    return optimized_spins, result.fun  # Return spin configuration and ground state energy


# Example parameter set (J1, J2, J3, G1, Dx, H)
p = [1.0, 0.5, 0.3, 0.2, 0.1, 0.0]
ground_state_spins, ground_state_energy = find_ground_state(p)

print("Ground state spin configuration:")
print(ground_state_spins)
print("Ground state energy:", ground_state_energy)
