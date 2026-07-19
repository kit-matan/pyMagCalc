"""Regression tests for the 2026-07 code-review fixes.

Each test pins a bug that produced silently wrong output:

  * a Hamiltonian parameter whose name starts with "c" was counted as a boson
    operator by the LSWT degree filter, silently DELETING every term carrying it;
  * the numerical dispersion cache key ignored the Ewald dipole_dipole spec, so
    toggling/editing it served stale cached results;
  * update_hamiltonian_params rejected the object's own flat parameter list for
    any model with a vector parameter (round-trip failure).
"""
import copy

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel


def _fm_chain_config(param_name):
    """1-site ferromagnetic chain along a, J named `param_name`."""
    return {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 0.5}],
        },
        "parameters": {param_name: -1.0},
        "parameter_order": [param_name],
        "interactions": {
            "heisenberg": [
                {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": param_name},
                {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": param_name},
            ]
        },
        "magnetic_structure": {
            "enabled": True,
            "type": "pattern",
            "pattern_type": "ferromagnetic",
            "direction": [0, 0, 1],
        },
    }


def _dispersion(param_name, tmp_path):
    """Build via the CONFIG path (config_filepath), where parameter symbols keep
    their user-facing names -- the path on which a 'c*' name collided with the
    boson-operator name prefix. (The spin_model_module path renames them p0...)
    """
    cfg = copy.deepcopy(_fm_chain_config(param_name))
    cfg_file = tmp_path / f"chain_{param_name}.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    calc = mc.MagCalc(config_filepath=str(cfg_file), cache_mode="none")
    B = 2 * np.pi / 3.0
    qs = np.array([[f * B, 0.0, 0.0] for f in (0.1, 0.25, 0.5)])
    res = calc.calculate_dispersion(qs, serial=True)
    return calc, np.asarray(res.energies)


def test_parameter_named_c_star_is_not_treated_as_boson(tmp_path):
    """A parameter named 'chi' must give the SAME spectrum as one named 'J1'.

    The old filter identified bosons by the name prefix 'c', so every term
    carrying 'chi' was counted as degree 3 and silently dropped -- the FM chain
    came out with a flat zero spectrum.
    """
    _, e_ref = _dispersion("J1", tmp_path)
    _, e_chi = _dispersion("chi", tmp_path)
    assert np.nanmax(e_ref) > 0.1, "reference FM chain must disperse"
    assert np.allclose(e_chi, e_ref, atol=1e-8), (
        f"parameter named 'chi' changed the spectrum: {e_chi} vs {e_ref}"
    )


def test_numerical_cache_key_tracks_ewald_spec(tmp_path):
    """The dispersion cache key must change when the dipole_dipole (Ewald)
    block changes -- it is added to H(q) numerically, outside HMat_sym."""
    calc, _ = _dispersion("J1", tmp_path)
    q_list = [np.array([0.1, 0.0, 0.0])]
    key_plain = calc._generate_numerical_cache_key(q_list, "dispersion")
    calc.sm.config["dipole_dipole"] = {"method": "ewald"}
    key_ewald = calc._generate_numerical_cache_key(q_list, "dispersion")
    assert key_plain != key_ewald


def test_fast_evaluator_refuses_ewald(tmp_path):
    """The compiled dispersion evaluator would silently drop the Ewald term."""
    calc, _ = _dispersion("J1", tmp_path)
    calc.sm.config["dipole_dipole"] = {"method": "ewald"}
    with pytest.raises(RuntimeError, match="ewald"):
        calc.compile_dispersion_evaluator()


def test_update_hamiltonian_params_roundtrips_flat_list_with_vector_param():
    """update_hamiltonian_params(calc.hamiltonian_params) must not raise for a
    model with a vector parameter (H_dir): the attribute is stored FLAT while
    the old check only accepted the nested shape."""
    cfg = _fm_chain_config("J1")
    cfg["parameters"].update({"H_mag": 0.0, "H_dir": [0, 0, 1]})
    cfg["parameter_order"] = ["J1", "H_mag", "H_dir"]
    gm = GenericSpinModel(copy.deepcopy(cfg))
    thetas, phis = gm.generate_magnetic_structure()
    gm.set_magnetic_structure(thetas, phis)
    calc = mc.MagCalc(
        spin_model_module=gm,
        spin_magnitude=0.5,
        hamiltonian_params=[-1.0, 0.0, [0, 0, 1]],
        cache_file_base="test_review_vec",
        cache_mode="none",
    )
    flat = list(calc.hamiltonian_params)
    assert len(flat) == 5  # J1, H_mag, H_dir x3 -- stored flat
    calc.update_hamiltonian_params(flat)          # flat round-trip
    assert calc.hamiltonian_params == [float(x) for x in flat]
    calc.update_hamiltonian_params([-1.0, 0.0, [0, 0, 1]])  # nested still works
    assert calc.hamiltonian_params == [float(x) for x in flat]
