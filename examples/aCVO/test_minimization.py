
import os
import sys
import numpy as np
import yaml
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import magcalc as mc
try:
    import examples.aCVO.spin_model_hc as sm_hc
except ImportError:
    # Try local import if running from directory
    sys.path.append(current_dir)
    import spin_model_hc as sm_hc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMinimization")

def test_aCVO_minimization():
    logger.info("Testing generalized energy minimization for aCVO (H//c)...")
    
    # 1. Load Configuration
    config_file = os.path.join(current_dir, 'config_cvo_hc.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    model_params = config['model_params']
    # Extract params in order expected by spin_model_hc: J1, J2, J3, G1, Dx, Dy, D3, H
    # Note: config_cvo_hc.yaml in file viewing only showed J1..Dx, H.
    # But spin_model_hc.py expects 8 params including Dy, D3.
    # Let's check config_cvo_hc.yaml content again from memory or re-read?
    # I read it in step 21.
    # 1: model_params:
    # 2:   S: 0.5
    # 3:   J1: 2.49
    # 4:   J2: 2.7888
    # ...
    # 7:   Dx: 2.67
    # 8:   Dy: 0.0
    # 9:   D3: 0.0
    # 10:  H: 20.0 
    # Yes, it has Dy, D3.
    
    p_values = [
        model_params['J1'],
        model_params['J2'],
        model_params['J3'],
        model_params['G1'],
        model_params['Dx'],
        model_params['Dy'],
        model_params['D3'],
        model_params['H']
    ]
    S_val = model_params['S']
    
    logger.info(f"Loaded params: {p_values}")
    
    # 2. Initialize MagCalc
    # distinct cache base to avoid conflicts
    cache_base = 'test_min_cvo_hc' 
    # Use 'w' to ensure we generate symbols tailored to this model (though gen_HM handles symbolic params)
    # MagCalc will try to run mpr() which might complain about missing cached thetas, but it falls back to defaults.
    calc = mc.MagCalc(
        spin_magnitude=S_val,
        hamiltonian_params=p_values,
        spin_model_module=sm_hc,
        cache_file_base=cache_base,
        cache_mode='w' 
    )
    
    # 3. Reference Result (from spin_model_hc's optimized logic)
    logging.info("Running reference optimization (spin_model_hc specific)...")
    # This uses the fast vectorized code in spin_model_hc
    ref_thetas, ref_energy = sm_hc.find_ground_state_orientations(p_values, S_val, sm_hc)
    
    if ref_thetas is not None:
        ref_thetas_deg = np.degrees(ref_thetas)
        logger.info(f"Reference Energy: {ref_energy:.6f} meV")
        logger.info(f"Reference Mean Theta: {np.mean(ref_thetas_deg):.4f} deg")
        logger.info(f"Reference Canting (90-theta): {np.mean(90 - ref_thetas_deg):.4f} deg")
    else:
        logger.error("Reference optimization failed!")
        
    # 4. New Generalized Optimization
    logging.info("Running new generalized optimization (MagCalc.minimize_energy)...")
    # We start with random or default guess. Reference used pi/2 - perturbation.
    nspins = len(sm_hc.AL_SPIN_PREFERENCE)
    x0 = np.zeros(2 * nspins)
    # Interleave theta, phi
    for i in range(nspins):
        x0[2*i] = np.pi/2.0 - 0.05 # Theta guess
        # Try to match the preferred phi to see if we converge to same state easily
        # phi preference: 1 -> 0, -1 -> pi
        pref = sm_hc.AL_SPIN_PREFERENCE[i]
        x0[2*i+1] = 0.0 if pref == 1 else np.pi # Phi guess
        
    res = calc.minimize_energy(params=p_values, x0=x0)
    
    if res.success:
        logger.info(f"New Optimization Success! Energy: {res.fun:.6f} meV")
        
        # Extract results
        new_thetas = res.x[0::2]
        new_phis = res.x[1::2]
        
        new_thetas_deg = np.degrees(new_thetas)
        new_phis_deg = np.degrees(new_phis) % 360 # Normalize to 0-360
        
        logger.info(f"New Mean Theta: {np.mean(new_thetas_deg):.4f} deg")
        logger.info(f"New Canting (90-theta): {np.mean(90 - new_thetas_deg):.4f} deg")
        
        # Check agreement
        energy_diff = abs(res.fun - ref_energy)
        if energy_diff < 1e-4:
            logger.info("PASS: Generic minimization matches reference energy!")
        else:
            logger.warning(f"FAIL: Energy difference {energy_diff:.6f} meV is too large.")
            
        # Check phi consistency
        # We expect phi to be roughly 0 or 180 (0 or 3.14)
        for i, phi in enumerate(new_phis):
            # Normalize phi to [-pi, pi] for easier check vs 0/pi
            phi_norm = (phi + np.pi) % (2 * np.pi) - np.pi
            # Check if close to 0 or pi
            dist_to_0 = abs(phi_norm)
            dist_to_pi = abs(abs(phi_norm) - np.pi)
            if min(dist_to_0, dist_to_pi) > 0.1: # Allow some deviation if generalized allows it
                 logger.warning(f"Spin {i} phi={phi:.4f} ({np.degrees(phi):.1f} deg) deviates from collinear expectation.")
                 
    else:
        logger.error(f"New optimization failed: {res.message}")

if __name__ == "__main__":
    test_aCVO_minimization()
