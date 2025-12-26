import numpy as np
import os

def compare():
    f1_disp = 'disp_data.npz' # Designer
    f2_disp = 'CVO_propagation_disp_data.npz' # Propagation
    
    if not os.path.exists(f1_disp):
        print(f"File {f1_disp} missing.")
        return
    if not os.path.exists(f2_disp):
        print(f"File {f2_disp} missing.")
        return
        
    try:
        d1 = np.load(f1_disp)['energies']
        d2 = np.load(f2_disp)['energies']
        
        # d1 shape (N_q, N_modes).
        # Check shapes
        if d1.shape != d2.shape:
             print(f"Shape mismatch: {d1.shape} vs {d2.shape}")
        else:
             diff_max = np.nanmax(np.abs(d1 - d2))
             print(f"Dispersion Max Difference: {diff_max}")
             
    except Exception as e:
        print(f"Dispersion comparison error: {e}")

    f1_sqw = 'sqw_data.npz'
    f2_sqw = 'CVO_propagation_sqw_data.npz'
    
    if os.path.exists(f1_sqw) and os.path.exists(f2_sqw):
        try:
            s1 = np.load(f1_sqw)['intensities']
            s2 = np.load(f2_sqw)['intensities']
            
            diff_max = np.nanmax(np.abs(s1 - s2))
            print(f"S(Q,w) Intensity Max Difference: {diff_max}")
        except Exception as e:
             print(f"S(Q,w) comparison error: {e}")
    else:
         print("S(Q,w) files missing.")

if __name__ == "__main__":
    compare()
