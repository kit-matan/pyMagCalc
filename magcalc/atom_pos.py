# get atomic position from a .cif file
import read_cif as rc

atoms = rc.read_cif_file("KFe3J.cif")
print(atoms.cell.cellpar())
fe_positions = rc.get_atom_positions_in_unit_cell(atoms, "Fe")
fe_positions.sort(key=lambda pos: pos[2])  # Sort the positions based on the z-axis value
min_z = fe_positions[0][2]  # Get the smallest z-axis value

# Filter the Fe positions to only include those with the smallest z-axis value
fe_positions_planar = [pos for pos in fe_positions if pos[2] == min_z]

fe_positions_outside = rc.get_atom_positions_outside_unit_cell(atoms, fe_positions)

# Filter the Fe positions to only include those with the smallest z-axis value
fe_positions_outside_planar = [pos for pos in fe_positions_outside if pos[2] == min_z]

# the nearest neighbor distances up to the 5th nearest neighbor in the kagome plane
distances = rc.get_nearest_neighbor_distances(fe_positions_planar, fe_positions_outside_planar, 5)
print(distances)

# scattered plot of the atomic positions
rc.plot_atom_positions(fe_positions_planar, fe_positions_outside_planar)
