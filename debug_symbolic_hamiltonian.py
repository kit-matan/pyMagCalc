
import yaml
import sympy as sp
import numpy as np
import os
import sys

# Import Generic Model
from generic_model import GenericSpinModel

def main():
    print("--- Debugging Symbolic Hamiltonian ---")
    config_file = "KFe3J/KFe3J_declarative.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = os.path.dirname(os.path.abspath(config_file))
    sm = GenericSpinModel(config, base_path=base_path)
    
    # Create Symbols like MagCalc
    # Parameters: J1, J2, Dy, Dz, H
    J1, J2, Dy, Dz, H = sp.symbols('J1 J2 Dy Dz H')
    params_sym = [J1, J2, Dy, Dz, H]
    
    # Spins: S1..S27 (OUC)
    N_atoms_ouc = len(sm.atom_pos_ouc())
    print(f"Generating {N_atoms_ouc} spins...")
    Sxyz = []
    for i in range(N_atoms_ouc):
        sx = sp.Symbol(f"S{i}x", commutative=False)
        sy = sp.Symbol(f"S{i}y", commutative=False)
        sz = sp.Symbol(f"S{i}z", commutative=False)
        Sxyz.append([sx, sy, sz])
        
    print(f"Params: {params_sym}")
    print(f"Sxyz: {Sxyz}")
    
    # Call spin_interactions
    print("Calling spin_interactions...")
    Jex, DM = sm.spin_interactions(params_sym)
    
    # Check Jex content
    nz = 0
    sample = None
    rows, cols = Jex.shape
    for i in range(rows):
        for j in range(cols):
            if Jex[i, j] != 0:
                nz += 1
                sample = Jex[i,j]
    print(f"Jex Non-Zeros: {nz}")
    print(f"Jex Sample: {sample}")
    
    # Call Hamiltonian
    print("Calling Hamiltonian...")
    HM = sm.Hamiltonian(Sxyz, params_sym)
    
    print(f"HM Type: {type(HM)}")
    print(f"HM == 0? {HM == 0}")
    if HM != 0:
        print(f"HM Sample Term: {HM.args[0] if hasattr(HM, 'args') and len(HM.args)>0 else HM}")

if __name__ == "__main__":
    main()
