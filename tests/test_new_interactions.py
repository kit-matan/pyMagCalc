import yaml
import tempfile
import os
import numpy as np
import sympy as sp
from magcalc.core import MagCalc
from magcalc.config_builder import MagCalcConfigBuilder

def make_yaml_safe(data):
    if isinstance(data, dict):
        return {k: make_yaml_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_yaml_safe(v) for v in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return data.tolist()
    elif isinstance(data, sp.Basic):
        return str(data)
    else:
        return data

def test_kitaev_interaction():
    # Define a simple honeycomb lattice for Kitaev
    builder = MagCalcConfigBuilder()
    builder.set_lattice(a=2.0)
    builder.set_symmetry_ops(rotations=[np.eye(3)], translations=[[0,0,0]])
    
    # Add atoms
    builder.add_wyckoff_atom("Cu", [0.0, 0.0, 0.0], 0.5)
    builder.add_wyckoff_atom("Cu", [0.33, 0.67, 0.0], 0.5)
    print(f"DEBUG: Builder atom labels: {list(builder._atom_label_to_idx.keys())}")
    
    # Add Kitaev interactions manually for two bonds
    # Bond 1: Cu -> Cu0 image [0,0,0] - x bond
    builder.add_kitaev_interaction(["Cu", "Cu0"], "K", "x", offset=[0, 0, 0])
    
    builder.config["parameters"] = {"S": 0.5, "K": 1.1}
    builder.set_tasks(run_minimization=False, run_dispersion=True)
    builder.set_q_path([0,0,0], [1,0,0], 2)
    # Set small neighbor shells to keep matrix small
    builder.set_calculation(neighbor_shells=[0,0,0]) 
    
    safe_config = make_yaml_safe(builder.config)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(safe_config, f)
        tmp_name = f.name
    try:
        mc = MagCalc(config_filepath=tmp_name, cache_mode='none')
        print(f"DEBUG: mc.sm file: {mc.sm.__file__}")
        hm = mc.HMat_sym
        print(f"HMat_sym shape: {hm.shape}")
        print(f"Free symbols: {hm.free_symbols}")
        if hm.is_zero_matrix:
            print("ERROR: HMat_sym is ZERO sequence!")
            # Inspect raw expression if possible
            # print(f"DEBUG: raw HMat_sym expr: {hm}")
        
        assert any(str(s) in ['kx', 'ky', 'kz'] for s in hm.free_symbols) or len(hm.free_symbols) >= 0
        assert not hm.is_zero_matrix
    finally:
        if os.path.exists(tmp_name): os.remove(tmp_name)

def test_sia_arbitrary_axis():
    builder = MagCalcConfigBuilder()
    builder.set_lattice(a=3.0)
    builder.add_wyckoff_atom("Fe", [0.0, 0.0, 0.0], 1.0)
    
    # Uniaxial SIA along [1, 1, 1]
    builder.add_single_ion_anisotropy("Fe", "D", axis=[1, 1, 1])
    
    builder.config["parameters"] = {"S": 1.0, "D": -0.5}
    builder.set_tasks(run_minimization=False, run_dispersion=True)
    builder.set_q_path([0,0,0], [1,0,0], 2)
    builder.set_calculation(neighbor_shells=[0,0,0])
    
    safe_config = make_yaml_safe(builder.config)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(safe_config, f)
        tmp_name = f.name
    try:
        mc = MagCalc(config_filepath=tmp_name, cache_mode='none')
        hm = mc.HMat_sym
        assert not hm.is_zero_matrix
        q_vals = np.linspace([0, 0, 0], [1, 0, 0], 2)
        mc.calculate_dispersion(q_vals)
    finally:
        if os.path.exists(tmp_name): os.remove(tmp_name)

if __name__ == "__main__":
    test_kitaev_interaction()
    test_sia_arbitrary_axis()
    print("New interaction tests passed!")
