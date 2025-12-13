import numpy as np
from ase import io

def read_cif_file(file_path):
    """Read a CIF file and return the Atoms object."""
    atoms = io.read(file_path)
    return atoms


def get_atom_positions_in_unit_cell(atoms, atom_symbol):
    """Return the positions of Fe atoms in the unit cell."""
    atomic_positions = atoms.get_positions()
    atomic_symbols = atoms.get_chemical_symbols()
    atom_positions_all = [atomic_positions[i] for i, symbol in enumerate(atomic_symbols) if symbol == atom_symbol]
    return atom_positions_all


def get_atom_positions_outside_unit_cell(atoms, atom_positions):
    """Return the positions of Fe atoms outside the unit cell."""
    # add Fe atoms outside the unit cell
    lattice_constants = atoms.cell.cellpar()  # Get the lengths of the lattice constants

    # Calculate the angles between the lattice vectors
    a, b, c, alpha, beta, gamma = lattice_constants
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    volume = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    a_vector = np.array([a, 0, 0])
    b_vector = np.array([b * cos_gamma, b * sin_gamma, 0])
    c_vector = np.array([c * cos_beta, c * (cos_alpha * cos_gamma - cos_beta) / sin_gamma, volume / (a * b * sin_gamma)])  

    # Calculate the positions of Fe atoms outside the unit cell
    outside_unit_cell_atom_positions = []
    outside_unit_cell_atom_positions.extend(atom_positions)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    pass
                else:
                    position = atom_positions + i * a_vector + j * b_vector + k * c_vector
                    outside_unit_cell_atom_positions.extend(position)

    # outside_unit_cell_atom_positions = np.unique(outside_unit_cell_atom_positions, axis=0)

    return outside_unit_cell_atom_positions 


def get_nearest_neighbor_distances(atom_positions, outside_unit_cell_atom_positions, num_neighbors=3):
    """Return the distances to the nearest Fe neighbors."""
    # List the distance between the first three nearest neighnor Fe atoms and do not list the same distance twice
    distances = []
    for i in range(len(atom_positions)):
        for j in range(len(outside_unit_cell_atom_positions)):
            distance = round(np.linalg.norm(atom_positions[i] - outside_unit_cell_atom_positions[j]), 4)
            if distance not in distances:
                distances.append(distance)

    distances = sorted(distances)  # Sort the distances from low to high
    return distances[:num_neighbors]


def plot_atom_positions(atom_positions, outside_unit_cell_atom_positions):
    """Scatter plot of the atomic positions."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # make a projection to the xy plane
    
    
    ax.scatter(*zip(*atom_positions), label="Fe in unit cell")
    ax.scatter(*zip(*outside_unit_cell_atom_positions), label="Fe outside unit cell")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()