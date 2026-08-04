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

def _kitaev_hm(K_value):
    """Build the honeycomb Kitaev model through the BUILDER's own entry point
    (`add_kitaev_interaction`) and return its symbolic H(q)."""
    builder = MagCalcConfigBuilder()
    builder.set_lattice(a=2.0)
    builder.set_symmetry_ops(rotations=[np.eye(3)], translations=[[0, 0, 0]])
    builder.add_wyckoff_atom("Cu", [0.0, 0.0, 0.0], 0.5)
    builder.add_wyckoff_atom("Cu", [0.33, 0.67, 0.0], 0.5)
    # Bond 1: Cu -> Cu0 image [0,0,0] - x bond
    builder.add_kitaev_interaction(["Cu", "Cu0"], "K", "x", offset=[0, 0, 0])
    builder.config["parameters"] = {"S": 0.5, "K": K_value}
    builder.set_tasks(run_minimization=False, run_dispersion=True)
    builder.set_q_path([0, 0, 0], [1, 0, 0], 2)
    builder.set_calculation(neighbor_shells=[0, 0, 0])  # keep the matrix small

    safe_config = make_yaml_safe(builder.config)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(safe_config, f)
        tmp_name = f.name
    try:
        return MagCalc(config_filepath=tmp_name, cache_mode='none').HMat_sym
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_kitaev_interaction():
    """The builder's `add_kitaev_interaction` entry point reaches the Hamiltonian.

    This test used to assert

        assert any(str(s) in ['kx','ky','kz'] for s in hm.free_symbols) or len(...) >= 0
        assert not hm.is_zero_matrix

    whose first line is a tautology (`len(...) >= 0` is always true), leaving only
    "H is not identically zero" -- which a wrong axis, wrong sign or wrong magnitude
    all pass. Correctness of the Kitaev term itself is pinned in tests/test_kitaev.py
    against its exact interaction_matrix equivalent; what belongs HERE is that this
    particular entry point works, and that K reaches H with the right weight.
    """
    hm = _kitaev_hm(1.1)
    assert any(str(s) in ['kx', 'ky', 'kz'] for s in hm.free_symbols)
    assert not hm.is_zero_matrix

    # H is LINEAR in K (a single bilinear bond term), so doubling K doubles every
    # entry. An exact identity -- no golden number. It catches K being ignored
    # (hm2 would equal hm) or entering nonlinearly; it does NOT catch a uniform
    # wrong prefactor, which cancels in the ratio -- that is what the
    # interaction_matrix identity in tests/test_kitaev.py is for.
    hm2 = _kitaev_hm(2.2)
    diff = sp.simplify(sp.expand(sp.Matrix(hm2) - 2 * sp.Matrix(hm)))
    assert diff.is_zero_matrix, diff


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
