import sys
import os
import numpy as np
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from magcalc.config_builder import MagCalcConfigBuilder

def test_symmetry_logic():
    print("=== Testing Symmetry Logic ===")
    
    # 1. Setup Builder with Simple Cubic / Tetragonal (P4mm, #99)
    # Atoms at (0,0,0). a=4, c=4.
    builder = MagCalcConfigBuilder()
    builder.set_lattice(a=4.0, c=5.0, space_group=123) # P4/mmm (123) - high symmetry
    # P4/mmm has inversion center at bond midpoints usually.
    
    builder.add_wyckoff_atom("Cu", [0, 0, 0], 1.0) # 1a site
    
    print(f"Atoms in Unit Cell: {len(builder.atoms_uc)}")
    print(f"Space Group Ops: {len(builder.symmetry_ops['rotations'])}")

    # 2. Analyze Bond Symmetry
    # Nearest neighbor should be 4.0 (along x, y).
    max_dist = 4.1
    orbits = builder.analyze_bond_symmetry(max_distance=max_dist)
    
    print(f"\nFound {len(orbits)} orbits up to {max_dist} A.")
    
    ref_bond = None
    for i, orb in enumerate(orbits):
        rep = orb["representative"]
        dist = orb["distance"]
        mult = orb["multiplicity"]
        print(f"Orbit {i}: Dist={dist:.4f}, Mult={mult}, Rep={rep['atom_i']}->{rep['atom_j']} off={rep['offset']}")
        
        if abs(dist - 4.0) < 0.1:
            ref_bond = orb
            
    if not ref_bond:
        print("Error: Could not find NN bond at 4.0 A")
        return

    # 3. Get Constraints
    print("\n--- Constraints for NN Bond ---")
    constraints = builder.get_bond_constraints(ref_bond)
    
    print(f"Little Group Size: {constraints['little_group_size']}")
    print("Symbolic Matrix:")
    for row in constraints["symbolic_matrix"]:
        print(row)
    print("Free Parameters:", constraints["free_parameters"])
    print("Is Centrosymmetric:", constraints["is_centrosymmetric"])
    
    # DEBUG: Check if we can see why it failed
    # builder.get_bond_constraints prints nothing, let's copy the logic here or trust the fix.
    # Actually, inspecting the rotations might help.
    
    print("\nDebug: Inspecting Ops...")
    # Access private or re-derive?
    # Let's just trust that I will fix the builder next.
    pass

if __name__ == "__main__":
    test_symmetry_logic()
