import sys
import os
import numpy as np

# Add local dir to path
sys.path.append(os.getcwd())

try:
    from KFe3J import spin_model
except ImportError:
    # Try direct import if running from root
    sys.path.append(os.path.join(os.getcwd(), 'KFe3J'))
    import spin_model

def get_offset(pos_diff, uc):
    # Solve pos_diff = i*a + j*b + k*c
    # In 2D case of spin_model: a, b
    # Solve linear system
    coeffs = np.linalg.solve(uc.T, pos_diff)
    return np.round(coeffs).astype(int)

def main():
    uc = spin_model.unit_cell() # (3, 3)
    apos = spin_model.atom_pos()
    apos_ouc = spin_model.atom_pos_ouc()
    
    print(f"Number of atoms in UC: {len(apos)}")
    print(f"Number of atoms OUC: {len(apos_ouc)}")
    
    # Manual indices from spin_model.py
    # DMmat[0, 1] = -DMvec2
    # DMmat[0, 2] = -DMvec1
    # DMmat[0, 7] = -DMvec2
    # DMmat[0, 14] = -DMvec1
    # DMmat[1, 0] = DMvec2
    # DMmat[1, 14] = DMvec3
    # DMmat[1, 21] = DMvec2
    # DMmat[1, 23] = DMvec3
    # DMmat[2, 0] = DMvec1
    # DMmat[2, 7] = -DMvec3
    # DMmat[2, 15] = DMvec1
    # DMmat[2, 16] = -DMvec3
    
    manual_indices = [
        (0, 1), (0, 2), (0, 7), (0, 14),
        (1, 0), (1, 14), (1, 21), (1, 23),
        (2, 0), (2, 7), (2, 15), (2, 16)
    ]
    
    print("\n--- Mapping Manual Indices to Offsets ---")
    for i, j in manual_indices:
        pos_i = apos_ouc[i] # Should be same as apos[i] for i<3
        pos_j = apos_ouc[j]
        
        diff = pos_j - pos_i
        
        # Determine closest UC atom for j
        # j_uc_index = j % 3 ? No, depends on construction order.
        # spin_model construction:
        # [uc_atoms] + ... loop i,j ... loop k
        # So j corresponds to atom k = (j-3) % 3? No.
        # Let's find minimizing distance to atom_k + offset
        
        best_k = -1
        best_offset = None
        min_dist = 1e9
        
        for k in range(len(apos)):
            # Check displacement from atom k
            # R_j = R_k + T
            # T = R_j - R_k
            T = pos_j - apos[k]
            # Check if T is integer combo of unit vectors
            offset_float = np.linalg.solve(uc.T, T)
            offset_round = np.round(offset_float).astype(int)
            dist_check = np.linalg.norm(T - offset_round @ uc)
            
            if dist_check < 1e-5:
                # Found match
                if abs(dist_check) < min_dist:
                    min_dist = dist_check
                    best_k = k
                    best_offset = offset_round
                    
        print(f"Index pair ({i}, {j}): Atom_{i} -> Atom_{best_k} + Offset {best_offset}")

if __name__ == "__main__":
    main()
