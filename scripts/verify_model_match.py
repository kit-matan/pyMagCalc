
import numpy as np
import sympy as sp
import yaml
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import Manual Model - Assuming it is in examples/KFe3J relative to root, so ../examples/KFe3J from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/KFe3J')))

import spin_model as manual_model

# Import Generic Model
from magcalc.generic_model import GenericSpinModel

def main():
    # Test Parameters
    # [J1, J2, Dy, Dz, H]
    p_val = [3.23, 0.11, 0.2, 0.2, 0.0]
    
    # --- Manual Model ---
    print("--- Calculating Manual Interactions ---")
    Jex_man_sym, DM_man_sym = manual_model.spin_interactions(p_val)
    
    # Convert to numerical numpy arrays
    Jex_man = np.array(Jex_man_sym, dtype=float)
    
    # DM is matrix of 1x3 Matrices. Convert to (N, N_ouc, 3) array
    rows, cols = DM_man_sym.shape
    DM_man = np.zeros((rows, cols, 3))
    for i in range(rows):
        for j in range(cols):
            vec = DM_man_sym[i, j] 
            try:
                if hasattr(vec, 'shape'):
                     v = np.array(vec, dtype=float).flatten()
                     DM_man[i, j] = v
            except Exception as e:
                pass

    # --- Declarative Model ---
    print("--- Calculating Declarative Interactions ---")
    config_file = "KFe3J/KFe3J_declarative.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = os.path.dirname(os.path.abspath(config_file))
    sm = GenericSpinModel(config, base_path=base_path)
    
    atoms_ouc_man = np.array(manual_model.atom_pos_ouc())
    atoms_ouc_decl = sm.atom_pos_ouc() 
    
    scaling_factor = 1.0 # Unit scaling
    atoms_ouc_man_scaled = atoms_ouc_man * scaling_factor
    
    print(f"Manual OUC count: {len(atoms_ouc_man)}")
    print(f"Decl OUC count: {len(atoms_ouc_decl)}")
    
    # Map Decl indices to Manual indices
    map_man_to_decl = {}
    
    for m_i, m_pos in enumerate(atoms_ouc_man_scaled):
        dists = np.linalg.norm(atoms_ouc_decl - m_pos, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist < 1e-3:
            map_man_to_decl[m_i] = min_idx
        else:
            print(f"WARNING: Manual Atom {m_i} at {m_pos} has no match in Decl. Min dist: {min_dist}")

    print(f"Mapped {len(map_man_to_decl)} atoms from Manual to Decl.")
    
    Jex_decl_sym, DM_decl_sym = sm.spin_interactions(p_val)
    Jex_decl_full = np.array(Jex_decl_sym, dtype=float)
    r_d, c_d = Jex_decl_full.shape # Use shape from numpy array
    
    DM_decl_full = np.zeros((r_d, c_d, 3))
    for i in range(r_d):
        for j in range(c_d):
            entry = DM_decl_sym[i][j]
            if entry is not None:
                try:
                    v = np.array(entry, dtype=float).flatten()
                    DM_decl_full[i, j] = v
                except:
                    pass

    Jex_decl_mapped = np.zeros_like(Jex_man)
    DM_decl_mapped = np.zeros_like(DM_man)
    
    for m_j in range(len(atoms_ouc_man)):
        if m_j in map_man_to_decl:
            d_j = map_man_to_decl[m_j]
            Jex_decl_mapped[:, m_j] = Jex_decl_full[:, d_j]
            DM_decl_mapped[:, m_j] = DM_decl_full[:, d_j]
            
    print("\n--- Comparing Jex ---")
    diff_j = np.linalg.norm(Jex_man - Jex_decl_mapped)
    print(f"Jex Difference (Mapped): {diff_j:.6e}")
    if diff_j > 1e-6:
         print(f"Max Jex Decl: {np.max(np.abs(Jex_decl_mapped))}")
    
    print("\n--- Comparing DM ---")
    diff_dm = np.linalg.norm(DM_man - DM_decl_mapped)
    print(f"DM Difference (Mapped): {diff_dm:.6e}")
    
    mapped_decl_indices = set(map_man_to_decl.values())
    extra_energy = 0.0
    for d_j in range(len(atoms_ouc_decl)):
        if d_j not in mapped_decl_indices:
            extra_energy += np.sum(np.abs(Jex_decl_full[:, d_j]))
            extra_energy += np.sum(np.linalg.norm(DM_decl_full[:, d_j], axis=1))

    print(f"Extra Interaction Energy in Decl (outside mapped atoms): {extra_energy:.6e}")

if __name__ == "__main__":
    main()
