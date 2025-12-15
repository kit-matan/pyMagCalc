import pytest
import numpy as np
import os
import sys

# Ensure magcalc is in path (it should be if installed or running from root)
# Add KFe3J example directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
examples_dir = os.path.join(project_root, "examples", "KFe3J")
if examples_dir not in sys.path:
    sys.path.append(examples_dir)

try:
    import spin_model
except ImportError:
    pytest.skip("KFe3J spin_model not found", allow_module_level=True)

from magcalc.core import MagCalc

def test_kfe3j_integration():
    """
    Integration test using KFe3J example.
    Verifies that the refactored code produces valid dispersion results.
    """
    # Parameters from standard KFe3J usage
    S = 2.5
    # J1, J2, Dy, Dz, H
    params = [3.23, 0.11, 0.218, -0.195, 0.0]  
    
    cache_base = "test_integration_kfe3j"
    
    # Use 'w' mode to force regeneration and test the full pipeline (symbolic + numerical)
    # Ideally use a temp dir for cache, but MagCalc writes to fixed ../cache location currently.
    # We accept this for now.
    
    mc = MagCalc(
        spin_magnitude=S, 
        hamiltonian_params=params, 
        spin_model_module=spin_model, 
        cache_mode="w", 
        cache_file_base=cache_base
    )
    
    # M point
    q_vec = np.array([2 * np.pi / np.sqrt(3), 0, 0])
    
    res = mc.calculate_dispersion([q_vec])
    
    assert res is not None
    energies = res.energies[0]
    
    print(f"Calculated energies at M: {energies}")
    
    # Assertions
    assert len(energies) == mc.nspins
    assert not np.any(np.isnan(energies)), "Energies contain NaNs"
    
    # Basic physics check: energies should be positive for stable ground state
    # (Allowing for small numerical noise around 0 for Goldstone modes)
    assert np.all(energies > -1e-4), f"Found negative energies: {energies}"

    # Check S(q,w) execution
    res_sqw = mc.calculate_sqw([q_vec])
    assert res_sqw is not None
    assert len(res_sqw.intensities[0]) == mc.nspins
    assert not np.any(np.isnan(res_sqw.intensities)), "Intensities contain NaNs"
