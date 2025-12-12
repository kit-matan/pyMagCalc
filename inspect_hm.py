
import pickle
import sympy as sp
import numpy as np
import sys

def inspect_hm(filename):
    print(f"Loading {filename}...")
    try:
        with open(filename, 'rb') as f:
            HMat = pickle.load(f)
        
        print(f"Type: {type(HMat)}")
        # Sympy Matrix
        if hasattr(HMat, 'shape'):
             rows, cols = HMat.shape
             print(f"Shape: ({rows}, {cols})")
             
             nonzero = 0
             sample = None
             for i in range(rows):
                 for j in range(cols):
                     if HMat[i, j] != 0:
                         nonzero += 1
                         if sample is None: sample = HMat[i, j]
             
             print(f"Non-zero entries: {nonzero}")
             if sample:
                 print(f"Sample Entry: {sample}")
             else:
                 print("matrix is ALL ZEROS.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_hm("/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/research/magcalc/pyMagCalc_cache/symbolic_matrices/KFe3J_decl_final_cache_HM.pck")
