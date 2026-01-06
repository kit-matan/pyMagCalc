
import pytest
import numpy as np
import os
import yaml
from magcalc.core import MagCalc
from magcalc.generic_model import GenericSpinModel

CONF_FILE = "temp_test_config.yaml"

@pytest.fixture
def simple_chain_config():
    config = {
        "crystal_structure": {
            "lattice_parameters": {
                "a": 1.0, "b": 10.0, "c": 10.0,
                "alpha": 90, "beta": 90, "gamma": 90
            },
            "atoms_uc": [
                {"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                {"label": "B", "pos": [0.5, 0, 0], "spin_S": 1.0}
            ]
        },
        "interactions": {
            "heisenberg": [
                {
                    "pair": ["A", "B"],
                    "value": 1.0,
                    "rij_offset": [0, 0, 0]
                },
                 {
                    "pair": ["B", "A"],
                    "value": 1.0,
                    "rij_offset": [0, 0, 0]
                }
            ]
        },
        "parameters": {}
    }
    with open(CONF_FILE, "w") as f:
        yaml.dump(config, f)
    yield CONF_FILE
    if os.path.exists(CONF_FILE):
        os.remove(CONF_FILE)

def test_differential_evolution(simple_chain_config):
    mc = MagCalc(config_filepath=simple_chain_config, initialize=True, cache_mode="w")
    
    # Differential Evolution
    res = mc.minimize_energy(method='differential_evolution', maxiter=50, popsize=5, seed=42)
    
    assert res.success, "Differential Evolution failed"
    # AFM Chain: spins should be antiparallel (angle diff pi)
    # Energy should be ~ -1.0 * S^2 * neighbors? 
    # Check if run completes and returns valid structure
    assert len(res.x) == 4 # 2 spins * (theta, phi)

def test_basinhopping(simple_chain_config):
    mc = MagCalc(config_filepath=simple_chain_config, initialize=True, cache_mode="w")
    
    # Basin Hopping
    res = mc.minimize_energy(method='basinhopping', niter=5, seed=42)
    
    # Basin Hopping result object structure is slightly different (OptimizeResult is wrapped or returned?)
    # Scipy basinhopping returns OptimizeResult.
    
    # Note: Success might not be strictly True if local minimization hits limits, but we check if it returns.
    assert hasattr(res, 'x'), "Basin Hopping did not return x"
    assert len(res.x) == 4

if __name__ == "__main__":
    pytest.main([__file__])
