
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
    import examples.aCVO.spin_model as spin_model
except ImportError:
    # Try local import if running from directory
    sys.path.append(current_dir)
    import spin_model_hc as sm_hc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMinimization")

def test_aCVO_minimization():
    logger.info("Testing generalized energy minimization for aCVO...")
    
    # 1. Load Configuration
    # Use standard config.yaml
    config_file = os.path.join(current_dir, 'config.yaml')
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} not found. Using default parameters.")
        model_params = {}
    else:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model_params = config.get('cvo_model', {}).get('model_params', {})
        
    # Default parameters if config missing or empty
    J1 = model_params.get('J1', 2.49)
    J2 = model_params.get('J2_ratio', 1.12) * J1
    J3 = model_params.get('J3_ratio', 2.03) * J1
    G1 = model_params.get('G1', 0.28)
    Dx = model_params.get('Dx', 2.67)
    Dy = model_params.get('Dy', -2.0)
    D3 = model_params.get('D3', 0.0)
    
    # Test setting: H along c, 20 T
    H_mag = 20.0
    H_dir = [0, 0, 1]
    
    # New parameter format: [J1, J2, J3, G1, Dx, Dy, D3, H_dir, H_mag]
    p_values = [J1, J2, J3, G1, Dx, Dy, D3, H_dir, H_mag]
    S_val = model_params.get('S', 0.5)
    
    logger.info(f"Test params: {p_values}")
    
    # 2. Initialize MagCalc
    cache_base = 'test_min_cvo_unified' 
    
    # Initialize with 'auto' mode. 
    # Note: spin_model module is passed directly.
    try:
        calc = mc.MagCalc(
            spin_magnitude=S_val,
            hamiltonian_params=p_values,
            spin_model_module=spin_model,
            cache_file_base=cache_base,
            cache_mode='auto' 
        )
    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}")
        return

    # 3. Run Optimization
    logging.info("Running energy minimization...")
    nspins = calc.nspins
    x0 = np.zeros(2 * nspins)
    for i in range(nspins):
        x0[2*i] = np.pi/2.0 - 0.05 
        x0[2*i+1] = 0.0 
        
    res = calc.minimize_energy(x0=x0)
    
    if res.success:
        logger.info(f"Optimization Success! Energy: {res.fun:.6f} meV")
        new_thetas = res.x[0::2]
        new_thetas_deg = np.degrees(new_thetas)
        
        mean_theta = np.mean(new_thetas_deg)
        mean_canting = np.mean(90 - new_thetas_deg)
        
        logger.info(f"New Mean Theta: {mean_theta:.4f} deg")
        logger.info(f"New Mean Canting (90-theta): {mean_canting:.4f} deg")
        
        # Simple sanity check: Energy should be negative (usually)
        if res.fun < 0:
             logger.info("PASS: Energy is negative as expected.")
        else:
             logger.warning(f"Energy is positive ({res.fun:.4f}). Check if this is expected.")
             
    else:
        logger.error(f"Optimization failed: {res.message}")

if __name__ == "__main__":
    test_aCVO_minimization()
