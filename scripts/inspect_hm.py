
import pickle
import sympy as sp
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    # Used to hard-code a cache path inside a OneDrive copy of this tree that no
    # longer exists (and whose stale `magcalc/` could shadow this one -- see
    # magcalc/provenance.py). Take the .pck to inspect as an argument instead.
    if len(sys.argv) != 2:
        print("usage: python scripts/inspect_hm.py <path/to/..._HM.pck>\n"
              "  (caches live in pyMagCalc_cache/symbolic_matrices/ next to "
              "wherever you ran magcalc)")
        raise SystemExit(2)
    inspect_hm(sys.argv[1])
