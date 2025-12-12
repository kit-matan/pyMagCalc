
import numpy as np
from itertools import product

def unit_cell():
    """a unit cell for the kagome lattice"""
    va = np.array([np.sqrt(3) / 2, -1 / 2, 0])
    vb = np.array([0, 1, 0])
    vc = np.array([0, 0, 1])
    uc = [va, vb, vc]
    return np.array(uc)

def atom_pos():
    """atomic positions for the kagome lattice"""
    atom1 = np.array([0, 0, 0])
    atom2 = np.array([np.sqrt(3) / 4, -1 / 4, 0])
    atom3 = np.array([0, 1 / 2, 0])
    r_pos = [atom1, atom2, atom3]
    return np.array(r_pos)

def atom_pos_ouc():
    """atomic positions outside the unit cell REPLICATING spin_model.py LOGIC"""
    uc = unit_cell()
    apos = atom_pos()
    apos_len = len(apos)
    
    # Logic from spin_model.py
    r_pos_ouc = [apos[0], apos[1], apos[2]] + [
        apos[k] + i * uc[0] + j * uc[1]
        for i, j in product(range(-1, 2), repeat=2)
        if i != 0 or j != 0
        for k in range(apos_len)
    ]
    
    # Store origin info
    origin_info = []
    # indices 0,1,2 are UC
    origin_info.extend([(0, (0,0,0)), (1, (0,0,0)), (2, (0,0,0))])
    
    for i, j in product(range(-1, 2), repeat=2):
        if i != 0 or j != 0:
            for k in range(apos_len):
                origin_info.append((k, (i, j, 0)))
                
    return np.array(r_pos_ouc), origin_info

def main():
    target_indices = [1, 2, 7, 14, 0, 15, 16, 21, 23]
    
    _, origin_info = atom_pos_ouc()
    
    print("Mapping for manual indices:")
    for idx in target_indices:
        if idx < len(origin_info):
            k, offset = origin_info[idx]
            print(f"Index {idx} -> Atom {k} (in UC) + Offset {offset}")
        else:
            print(f"Index {idx} -> OUT OF BOUNDS")

if __name__ == "__main__":
    main()
