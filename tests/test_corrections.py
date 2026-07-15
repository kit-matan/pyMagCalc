"""1/S (LSWT) corrections: zero-point energy and ordered-moment reduction.

Validated against Sunny 0.8.1 AND the textbook S=1/2 square-lattice Heisenberg AFM:
    zero-point energy correction  dE = -0.157947 J/site
    ordered-moment reduction      dS =  0.1966       (<S^z> = S - dS)
"""
import logging
import warnings

import numpy as np
import pytest

import magcalc as mc
from magcalc.corrections import _colpa, compute_corrections
from magcalc.generic_model import GenericSpinModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def _build(cfg, S):
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="corr", hamiltonian_params=[])


def test_colpa_is_para_unitary():
    """T must satisfy T^dag g T = g (para-unitarity) and reproduce omega."""
    # a simple 1-mode BdG: A a^dag a + (B/2)(a^dag a^dag + a a), omega = sqrt(A^2-B^2)
    A, Bb = 2.0, 0.8
    H2 = np.array([[A, Bb], [Bb, A]], dtype=complex)
    omega, T = _colpa(H2)
    g = np.diag([1.0, -1.0])
    assert np.allclose(T.conj().T @ g @ T, g, atol=1e-10)
    assert np.allclose(omega, np.sqrt(A**2 - Bb**2), atol=1e-10)


def test_ferromagnet_has_zero_corrections():
    """The cleanest check of the whole machinery: for a ferromagnet the classical state
    IS the exact ground state (the magnon vacuum), so both corrections are identically
    zero. Any nonzero result is a convention/sign bug."""
    cfg = {
        "crystal_structure": {"lattice_vectors": [[3, 0, 0], [0, 3, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                            "spin_S": 1.0, "ion": "Fe2+"}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": -1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    r = compute_corrections(_build(cfg, 1.0), k_mesh=(20, 20, 1))
    assert abs(r.energy_correction_per_site) < 1e-9
    assert abs(r.moment_reduction[0]) < 1e-9


# sqrt(2)xsqrt(2) Neel cell of the square-lattice AFM (a1=(1,1), a2=(1,-1))
_AB = [[0, 0, 0], [-1, -1, 0], [0, -1, 0], [-1, 0, 0]]
_SQ_AFM = {
    "crystal_structure": {
        "lattice_vectors": [[1.0, 1.0, 0], [1.0, -1.0, 0], [0, 0, 10.0]],
        "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 0.5, "ion": "Cu2+"},
                     {"label": "B", "pos": [0.5, 0.5, 0.0], "spin_S": 0.5, "ion": "Cu2+"}]},
    "interactions": {"heisenberg":
        [{"pair": ["A", "B"], "rij_offset": o, "value": 1.0} for o in _AB]
        + [{"pair": ["B", "A"], "rij_offset": [-o[0], -o[1], 0], "value": 1.0} for o in _AB]},
    "parameters": {}, "parameter_order": [],
    "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                           "directions": [[0, 0, 1], [0, 0, -1]]},
}


def test_square_afm_energy_correction_matches_sunny_and_textbook():
    """S=1/2 square Heisenberg AFM: dE = -0.157947 J/site (Sunny + textbook). The energy
    converges fast even across the Goldstone cone."""
    r = compute_corrections(_build(_SQ_AFM, 0.5), k_mesh=(48, 48, 1))
    assert abs(r.energy_correction_per_site - (-0.157947)) < 1e-4, \
        r.energy_correction_per_site


def test_square_afm_moment_reduction_converges_to_textbook():
    """dS -> 0.1966. It converges SLOWLY (the 1/omega integrand near the gapless cone),
    so check the trend toward the known value rather than a single mesh."""
    S = 0.5
    ds = [compute_corrections(_build(_SQ_AFM, S), k_mesh=(n, n, 1)).moment_reduction[0]
          for n in (32, 64, 128)]
    assert ds[0] < ds[1] < ds[2]                    # monotone increase toward 0.1966
    assert abs(ds[-1] - 0.1966) < 0.005             # within 2.5% at 128^2
    assert all(d < 0.1966 for d in ds)              # approaches from below


def test_corrections_warn_on_a_non_ground_state(caplog):
    """If the reference structure is not a minimum, H(q) has imaginary modes and the
    corrections are meaningless -- warn rather than return a plausible number."""
    cfg = {
        "crystal_structure": {"lattice_vectors": [[3, 0, 0], [0, 3, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                            "spin_S": 1.0, "ion": "Fe2+"}]},
        # FM alignment with ANTIferromagnetic exchange -> not a minimum
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    with pytest.raises(ValueError, match="NOT a classical minimum"):
        compute_corrections(_build(cfg, 1.0), k_mesh=(8, 8, 1))


def test_runner_corrections_task(tmp_path):
    """`tasks: {corrections: true}` runs end to end and reports finite numbers."""
    import yaml

    from magcalc.runner import run_calculation

    cfg = dict(_SQ_AFM)
    cfg["calculation"] = {"cache_mode": "none"}
    cfg["tasks"] = {"minimization": False, "corrections": True}
    cfg["corrections"] = {"k_mesh": [24, 24, 1]}
    cfg["plotting"] = {"save_plot": False, "show_plot": False, "plot_structure": False}
    cfg["output"] = {"save_data": False}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_calculation(str(p))          # must not raise
