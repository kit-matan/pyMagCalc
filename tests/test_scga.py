"""SCGA — paramagnetic diffuse scattering (magcalc/scga.py).

Pinned to independent oracles, never a self-generated number:

  * the EXACT closed form for the classical Heisenberg chain: J(q)=2J cos q, the sum
    rule gives λ = √(4J² + (3kT/S²)²) and S(q) = 3kT/(λ+2J cos q) analytically;
  * Sunny 0.8.1's own SCGA (`src/SCGA/SCGA.jl`) on the square-lattice AND kagome
    (frustrated, 3 equivalent sublattices) antiferromagnets — λ and S(q) to 6 digits.
    The golden numbers below were produced by Sunny (scripts in the commit message);
    pyMagCalc reproduces `fourier_exchange_matrix`, the single-λ sum rule, and the
    `kT·pref†(λ+J)⁻¹pref` intensity with `ssf_perp` (apply_g, (2/3)Tr at q=0);
  * the sum rule (λ solve) and the high-T limit (S(q) → flat) as loud invariants.
"""
import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc import scga


def _chain(S=0.5, J=1.0):
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 10, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": J},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    return GenericSpinModel(cfg)


@pytest.mark.parametrize("kT", [2.0, 0.8, 0.3])
def test_chain_lambda_and_sq_closed_form(kT):
    """Classical Heisenberg chain: λ and S(q) match the exact closed form."""
    S, J = 0.5, 1.0
    m = _chain(S, J)
    lam, lam_min, resid, cache, mesh = scga.solve_lambda(m, [], kT, nq=2000)
    lam_exact = np.sqrt(4 * J**2 + (3 * kT / S**2)**2)
    assert abs(lam - lam_exact) < 1e-9 * lam_exact
    assert resid < 1e-10
    qs = np.array([[2 * np.pi * r, 0, 0] for r in (0.0, 0.25, 0.5)])
    I = scga.scga_intensities(m, [], qs, kT, cross_section="trace", lam=lam,
                              mesh=mesh, apply_g=False)
    Iex = [3 * kT / (lam_exact + 2 * J * np.cos(2 * np.pi * r)) for r in (0.0, 0.25, 0.5)]
    assert np.allclose(I, Iex, atol=1e-9)


def test_high_temperature_flattens_sq():
    """As kT grows the diffuse S(q) must flatten toward a q-independent paramagnet."""
    m = _chain(0.5, 1.0)
    qs = np.array([[2 * np.pi * r, 0, 0] for r in np.linspace(0, 0.5, 11)])
    spreads = []
    for kT in (0.3, 3.0, 30.0):
        I = scga.scga_intensities(m, [], qs, kT, cross_section="trace", nq=2000,
                                  apply_g=False)
        spreads.append((I.max() - I.min()) / I.mean())
    assert spreads[0] > spreads[1] > spreads[2]
    assert spreads[2] < 0.05


# ---- Sunny 0.8.1 SCGA golden numbers (produced in-repo; see commit message) --------
def _square(S=1.0, J=1.0):
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": o, "value": J}
            for o in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    return GenericSpinModel(cfg), np.array([[1., 0, 0], [0, 1, 0], [0, 0, 10]])


def test_square_lattice_matches_sunny():
    m, lv = _square()
    kT = 2.0
    lam, *_rest, cache, mesh = scga.solve_lambda(m, [], kT, nq=20)
    assert abs(lam - 6.683045740423269) < 1e-6
    B = 2 * np.pi * np.linalg.inv(lv).T
    sunny = {(0.0, 0.0): 1.497700224146571, (0.5, 0.0): 2.394117984741885,
             (0.5, 0.5): 5.963372058456183, (0.25, 0.1): 1.9274601042283566}
    for (h, k), sv in sunny.items():
        q = np.array([h, k, 0]) @ B
        I = scga.scga_intensities(m, [], q, kT, cross_section="perp", lam=lam, mesh=mesh)[0]
        assert abs(I - sv) < 1e-5, f"q=({h},{k}): {I} vs Sunny {sv}"


def _kagome(S=1.0):
    g = np.radians(120.0)
    lv = [[1., 0, 0], [np.cos(g), np.sin(g), 0], [0, 0, 10]]
    cfg = {"crystal_structure": {"lattice_vectors": lv,
            "atoms_uc": [{"label": "A", "pos": [0.5, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0, 0.5, 0], "spin_S": S},
                         {"label": "C", "pos": [0.5, 0.5, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.5,
                                             "value": "J1"}]},
        "parameters": {"J1": 1.0}, "parameter_order": ["J1"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    return GenericSpinModel(cfg), np.array(lv)


def test_kagome_matches_sunny():
    """Frustrated 3-sublattice kagome AFM: single-λ SCGA vs Sunny, exact."""
    m, lv = _kagome()
    kT = 1.5
    lam, *_rest, cache, mesh = scga.solve_lambda(m, [1.0], kT, nq=25)
    assert cache["N"] == 3
    assert abs(lam - 5.18498918857465) < 1e-6
    B = 2 * np.pi * np.linalg.inv(lv).T
    sunny = {(0.0, 0.0): 3.9194384730230265, (1 / 3., 1 / 3.): 6.429707876172291,
             (0.5, 0.0): 5.654670497871585, (2 / 3., 1 / 3.): 9.98101731125756,
             (0.5, 0.5): 9.849720595717331}
    for (h, k), sv in sunny.items():
        q = np.array([h, k, 0]) @ B
        I = scga.scga_intensities(m, [1.0], q, kT, cross_section="perp", lam=lam,
                                  mesh=mesh)[0]
        assert abs(I - sv) < 1e-4, f"q=({h},{k}): {I} vs Sunny {sv}"


# ---------------------------------------------------------------------------
# The single-λ theory's limit of validity
# ---------------------------------------------------------------------------

def _mixed_spin_chain(S_a=1.0, S_b=2.5, J=1.0):
    """Two-site chain with UNEQUAL spins: two genuinely inequivalent sublattices."""
    cfg = {"crystal_structure": {
               "lattice_vectors": [[3., 0, 0], [0, 9., 0], [0, 0, 9.]],
               "atoms_uc": [{"label": "A", "pos": [0., 0, 0], "spin_S": S_a},
                            {"label": "B", "pos": [0.5, 0, 0], "spin_S": S_b}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J},
               {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J},
               {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": J},
               {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": J}]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]}}
    return GenericSpinModel(cfg)


def test_inequivalent_sublattices_are_refused():
    """One λ cannot satisfy two sublattice sum rules, and SCGA must say so.

    GAP_STATUS #9 claimed this was refused; nothing refused it. A mixed-spin chain
    (S = 1 and S = 5/2) returned an S(q) whose TOTAL sum-rule residual was 3e-16 --
    a perfect-looking solve -- while the two PER-SITE constraints were violated by
    +262% and -42%. Nothing in the output hinted at it, and every other test in this
    file is a single symmetry class, where the total constraint implies each one.
    """
    m = _mixed_spin_chain()
    with pytest.raises(ValueError, match="PER SUBLATTICE"):
        scga.compute_scga(m, [], np.array([[0.1, 0, 0], [0.5, 0, 0]]), kT=2.0, nq=12)


def test_the_refusal_reports_the_actual_violation():
    """The measured residuals, so the message is evidence and not a category label."""
    m = _mixed_spin_chain()
    with pytest.raises(ValueError) as exc:
        scga.solve_lambda(m, [], 2.0, nq=12)
    msg = str(exc.value)
    assert "site 0 (S=1)" in msg and "site 1 (S=2.5)" in msg, msg
    assert "+262" in msg and "-42" in msg, msg

    # and the residuals themselves, which is what the guard is reading
    lam, _lam_min, _res, cache, mesh = scga.solve_lambda(
        m, [], 2.0, nq=12, sublattice_tol=float("inf"))
    r = scga.sublattice_sum_rule_residuals(m, [], 2.0, lam, cache, mesh)
    assert r == pytest.approx([2.625, -0.42], abs=1e-6)


def test_one_symmetry_class_satisfies_every_sublattice_rule_exactly():
    """The guard cannot fire on the models the theory IS valid for.

    Kagome: three sites, one symmetry class, so the single λ that solves the total
    sum rule solves each site's automatically -- to ~1e-16, ten orders of magnitude
    inside the 1e-6 tolerance. This is why the residual is a better criterion than
    the symmetry labelling: it tests the physics the theory needs.
    """
    m, _lv = _kagome()
    kT = 1.5
    lam, _lam_min, _res, cache, mesh = scga.solve_lambda(m, [1.0], kT, nq=12)
    r = scga.sublattice_sum_rule_residuals(m, [1.0], kT, lam, cache, mesh)
    assert np.max(np.abs(r)) < 1e-12, r
    assert np.max(np.abs(r)) < scga.SUBLATTICE_SUM_RULE_TOL
