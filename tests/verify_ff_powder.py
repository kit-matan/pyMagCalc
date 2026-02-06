import numpy as np
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import KFe3J example
examples_dir = os.path.join(project_root, "examples", "KFe3J")
if examples_dir not in sys.path:
    sys.path.append(examples_dir)

try:
    import spin_model
except ImportError:
    print("KFe3J spin_model not found. Skipping integration part.")
    sys.exit(0)

from magcalc.core import MagCalc
from magcalc.form_factors import get_j0

def verify_form_factor():
    print("\n--- Verifying Form Factor ---")
    S = 2.5
    params = [3.23, 0.11, 0.218, -0.195, 0.0]  
    
    mc = MagCalc(
        spin_magnitude=S, 
        hamiltonian_params=params, 
        spin_model_module=spin_model, 
        cache_mode="w", 
        cache_file_base="test_ff_verify"
    )
    
    # Q at low and high magnitudes
    q_low = np.array([0.1, 0, 0])
    q_high = np.array([2 * np.pi, 0, 0])
    
    res_low = mc.calculate_sqw([q_low])
    res_high = mc.calculate_sqw([q_high])
    
    if res_low and res_high:
        int_low = np.sum(res_low.intensities[0])
        int_high = np.sum(res_high.intensities[0])
        
        # Calculate expected form factor ratio
        # KFe3J has Fe3+ (as per my default fallback and common usage)
        ff_low = get_j0("Fe3+", np.linalg.norm(q_low))
        ff_high = get_j0("Fe3+", np.linalg.norm(q_high))
        
        print(f"Intensity at low Q ({np.linalg.norm(q_low):.2f}): {int_low:.4e}")
        print(f"Intensity at high Q ({np.linalg.norm(q_high):.2f}): {int_high:.4e}")
        print(f"Form factor j0(low Q): {ff_low:.4f}")
        print(f"Form factor j0(high Q): {ff_high:.4f}")
        
        # The ratio of intensities should roughly scale with |F(Q)|^2
        # (ignoring polarization factor changes and mode weight changes for now)
        # Actually, at high Q, intensity MUST be lower if form factor is applied.
        
        if int_high < int_low:
            print("SUCCESS: Intensity decreased at higher Q.")
        else:
            print("FAILURE: Intensity did not decrease at higher Q. Check form factor application.")

def verify_powder_average():
    print("\n--- Verifying Powder Average ---")
    S = 2.5
    params = [3.23, 0.11, 0.218, -0.195, 0.0]  
    
    mc = MagCalc(
        spin_magnitude=S, 
        hamiltonian_params=params, 
        spin_model_module=spin_model, 
        cache_mode="auto", 
        cache_file_base="test_powder_verify"
    )
    
    q_mags = [0.1, 1.0, 2.0]
    res_powder = mc.calculate_powder_average(q_mags, num_samples=10) # Small sample for speed
    
    if res_powder:
        print(f"Powder average calculated for {len(q_mags)} points.")
        for i, q_mag in enumerate(q_mags):
            avg_int = np.sum(res_powder.intensities[i])
            print(f"|Q| = {q_mag:.2f}, Avg Intensity = {avg_int:.4e}")
        print("SUCCESS: Powder average calculation completed.")
    else:
        print("FAILURE: Powder average calculation failed.")

if __name__ == "__main__":
    verify_form_factor()
    verify_powder_average()
