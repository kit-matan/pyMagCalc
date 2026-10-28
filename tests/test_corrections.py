"""1/S (LSWT) corrections: zero-point energy and ordered-moment reduction.

Validated against Sunny 0.8.1 AND the textbook S=1/2 square-lattice Heisenberg AFM:
    zero-point energy correction  dE = -0.157947 J/site
    ordered-moment reduction      dS =  0.1966       (<S^z> = S - dS)
"""
import logging

import numpy as np
import pytest

import magcalc as mc
from magcalc.corrections import _colpa, compute_corrections
from magcalc.generic_model import GenericSpinModel

# Module-scoped, NOT a bare warnings.filterwarnings("ignore"): that is a
# process-global side effect, so it silenced warnings for every test module
# collected after this one alphabetically -- which is most of them. The
# "zero pytest warnings" the suite claims was partly that.
pytestmark = pytest.mark.filterwarnings("ignore")
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
    cfg["corrections"] = {"k_mesh": [24, 24, 1], "n_threads": 2}
    cfg["plotting"] = {"save_plot": False, "show_plot": False, "plot_structure": False}
    cfg["output"] = {"save_data": True, "corrections_filename": "corr.npz"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_calculation(str(p))          # must not raise
    # the saved numbers are the direct call's, so `n_threads` reached the solver
    # without changing anything but the schedule
    saved = np.load(tmp_path / "corr.npz")
    ref = compute_corrections(_build(_SQ_AFM, 0.5), k_mesh=(24, 24, 1), n_threads=1)
    assert int(saved["n_kpoints"]) == 576
    assert abs(float(saved["energy_correction_per_site"])
               - ref.energy_correction_per_site) < 1e-13
    np.testing.assert_allclose(saved["moment_reduction"], ref.moment_reduction,
                               rtol=0, atol=1e-13)


# --- the batched mesh (2026-10-28) -------------------------------------------------
# `compute_corrections` now takes the mesh in chunks through `_chunk_sums`, which
# replaces the per-k general eigensolve with a similarity argument (g H2 = g K^dag K
# is similar to the Hermitian K g K^dag, so a successful Cholesky means real
# eigenvalues) and falls back to the original loop, kept as `_chunk_sums_per_k`, for
# any chunk where that argument does not apply. The textbook and ferromagnet tests
# above are the physics oracles and run through the new path; these pin that the
# two paths agree and that the fallback is taken exactly when it must be.

def _one_mode_stack(AB):
    """g*H2 for one-mode BdG blocks H2 = [[A, B], [B*, A]], omega = sqrt(A^2 - |B|^2)."""
    g = np.diag([1.0, -1.0])
    return np.array([g @ np.array([[A, B], [np.conj(B), A]], dtype=complex)
                     for A, B in AB])


def test_batched_chunk_matches_the_per_k_loop_on_a_supercell():
    from magcalc.corrections import _chunk_sums, _chunk_sums_per_k
    from magcalc.core import reciprocal_b_matrix

    cfg = {**_SQ_AFM, "crystal_structure": {**_SQ_AFM["crystal_structure"],
                                            "magnetic_supercell": [2, 2, 1]}}
    calc = _build(cfg, 0.5)
    N = int(calc.nspins)
    assert N == 8
    B = reciprocal_b_matrix(np.asarray(calc.sm.unit_cell(), dtype=float))
    qs = np.random.default_rng(1).random((200, 3)) @ B
    H = calc._build_h_stack(qs, 0.5)
    z1, d1, u1, f1, i1 = _chunk_sums(H, N)
    z0, d0, u0, f0, i0 = _chunk_sums_per_k(H, N)
    assert (u1, f1) == (u0, f0) == (200, 0)
    assert abs(z1 - z0) < 1e-12 * abs(z0)
    np.testing.assert_allclose(d1, d0, rtol=0, atol=1e-12)
    assert i1 == 0.0 and i0 < 1e-10


def test_batched_chunk_on_complex_blocks_matches_the_per_k_loop():
    """Complex anomalous terms, i.e. the phases a non-collinear structure gives."""
    from magcalc.corrections import _chunk_sums, _chunk_sums_per_k
    rng = np.random.default_rng(2)
    AB = [(2.0 + rng.random(), 0.9 * rng.random() * np.exp(2j * np.pi * rng.random()))
          for _ in range(50)]
    H = _one_mode_stack(AB)
    a, b = _chunk_sums(H, 1), _chunk_sums_per_k(H, 1)
    assert abs(a[0] - b[0]) < 1e-12 and a[2:4] == b[2:4] == (50, 0)
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-12)
    # and the closed form: omega = sqrt(A^2 - |B|^2), <a^dag a> = (A/omega - 1)/2
    om = np.array([np.sqrt(A**2 - abs(B)**2) for A, B in AB])
    assert abs(a[0] - np.sum(om - np.array([A for A, _ in AB]))) < 1e-12
    assert abs(a[1][0] - np.sum([(A / w - 1) / 2 for (A, _), w in zip(AB, om)])) < 1e-12


def test_a_chunk_with_an_unstable_k_takes_the_per_k_loop():
    """One |B| > A block: H2 is not positive definite and omega is imaginary there.
    The batched Cholesky fails for the whole chunk, which must then report exactly
    what the original loop reports -- the failure count AND |Im omega|."""
    from magcalc.corrections import _chunk_sums, _chunk_sums_per_k
    AB = [(2.0, 0.5), (1.0, 1.5), (3.0, 0.2j)]
    a, b = _chunk_sums(_one_mode_stack(AB), 1), _chunk_sums_per_k(_one_mode_stack(AB), 1)
    assert a[2:4] == b[2:4] == (2, 1)
    assert a[4] == b[4] and abs(a[4] - np.sqrt(1.5**2 - 1.0)) < 1e-12
    assert a[0] == b[0]


def test_a_non_hermitian_chunk_takes_the_per_k_loop():
    """The similarity argument needs H2 Hermitian; the old loop's eigensolve ran on
    the raw H. A non-Hermitian H2 must therefore get the old loop, bit for bit."""
    from magcalc.corrections import _chunk_sums, _chunk_sums_per_k
    H = _one_mode_stack([(2.0, 0.5), (2.5, 0.3)])
    H[0, 0, 1] += 1e-3
    a, b = _chunk_sums(H, 1), _chunk_sums_per_k(H, 1)
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def test_threads_and_chunk_size_do_not_change_the_result():
    calc = _build(_SQ_AFM, 0.5)
    ref = compute_corrections(calc, k_mesh=(24, 24, 1), n_threads=1, chunk_size=10**6)
    for nt, cs in [(1, 7), (4, 7), (4, 100), (None, None)]:
        r = compute_corrections(calc, k_mesh=(24, 24, 1), n_threads=nt, chunk_size=cs)
        assert r.n_kpoints == ref.n_kpoints == 576
        assert abs(r.energy_correction_per_site - ref.energy_correction_per_site) < 1e-14
        np.testing.assert_allclose(r.moment_reduction, ref.moment_reduction,
                                   rtol=0, atol=1e-14)


@pytest.mark.slow
def test_batched_path_on_kfe3j_matches_the_per_k_loop(tmp_path, monkeypatch):
    """A real non-collinear model with DM (KFe3J's q = 0 kagome state, 9 sites,
    complex H(q)): the batched sums equal the original loop's at random k."""
    import os

    import yaml

    import magcalc.corrections as C
    from magcalc.core import reciprocal_b_matrix
    from magcalc.runner import run_calculation

    src = os.path.join(os.path.dirname(__file__), "..", "examples", "materials",
                       "KFe3J", "config_kfe3j.yaml")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["tasks"] = {"minimization": True, "corrections": True}
    cfg["minimization"] = {"enabled": True, "method": "anneal", "num_starts": 2,
                           "n_sweeps": 500, "seed": 0}
    cfg["corrections"] = {"k_mesh": [4, 4, 4]}
    cfg["calculation"] = {"cache_mode": "none"}
    cfg["plotting"] = {"save_plot": False, "show_plot": False, "plot_structure": False}
    cfg["output"] = {"save_data": False}
    p = tmp_path / "kfe3j.yaml"
    p.write_text(yaml.safe_dump(cfg))

    seen = {}
    real = C.compute_corrections

    def spy(calc, **kw):
        seen["calc"] = calc
        return real(calc, **kw)

    monkeypatch.setattr(C, "compute_corrections", spy)
    run_calculation(str(p))
    calc = seen["calc"]
    N = int(calc.nspins)
    assert N == 9
    B = reciprocal_b_matrix(np.asarray(calc.sm.unit_cell(), dtype=float))
    H = calc._build_h_stack(np.random.default_rng(0).random((100, 3)) @ B,
                            float(calc.spin_magnitude))
    assert np.max(np.abs(H.imag)) > 1e-2            # genuinely complex
    a, b = C._chunk_sums(H, N), C._chunk_sums_per_k(H, N)
    assert a[2:4] == b[2:4] == (100, 0)
    assert abs(a[0] - b[0]) < 1e-12 * abs(b[0])
    np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-11)
