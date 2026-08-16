"""The CP^(N-1) Metropolis sampler must sample, not just run (OPEN_WORK item 13).

THE DEFECT. `sigma` -- the size of the proposed move on CP^(N-1) -- does not change
the stationary distribution, only how fast the chain reaches it. That makes a fixed
`sigma` formally harmless and practically decisive: at low temperature almost every
proposal from a fixed step is uphill by >> kT and is rejected, so the chain barely
moves and `sampled_correlations` returns the spectrum of whatever it started from.
Item 4 measured it end to end -- the SU(N) classical intensity relative to
`SUNModel.structure_factor` swung **0.30 -> 1.63** on `therm_sweeps`/`sigma` alone,
so the answer was set by the sampler defaults rather than by the physics, and both
ends of that range look perfectly plausible. It is also why item 4 had to pin the
SU(N) half of its work with the exact sum rule and grid independence rather than
against LSWT.

THE ORACLE IS THE SAMPLER'S OWN PARTITION FUNCTION, IN CLOSED FORM, and that is the
point: it pins the SAMPLER, with no spectrum, no LSWT and no reference code in the
way. For a DECOUPLED site with on-site operator A = diag(a_1..a_N), the coherent
state energy is E(Z) = sum_i a_i |z_i|^2, and the uniform (Fubini-Study) measure on
CP^(N-1) makes (|z_1|^2 .. |z_N|^2) uniform on the simplex. So

    Z(beta) = int_simplex exp(-beta sum_i a_i p_i) dp
            = sum_i exp(-beta a_i) / prod_{j != i} (beta a_j - beta a_i)

for distinct a_i, and <E> = -d ln Z / d beta -- exact, checked against a direct
2-D quadrature in `test_the_closed_form_partition_function_is_right`.

What the tests below assert, in order: the closed form is right; the adapted sampler
reproduces it; the answer no longer depends on where `sigma` started (the property
that was false); the un-adapted sampler at a bad `sigma` is measurably wrong (the
defect, as a number); and the diagnostic fires on exactly those runs rather than on
the good ones.
"""
import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun import dynamics as sd
from magcalc.sun.lswt import SUNModel

D_SIA, B_FIELD, S_SPIN = -0.8, 5.0, 1.0
# The on-site operator that produces: D*Sz^2 + gamma*mu_B*B*Sz on |m = +1, 0, -1>.
LEVELS = np.array([D_SIA + 2 * 5.788e-2 * B_FIELD, 0.0,
                   D_SIA - 2 * 5.788e-2 * B_FIELD])


def _decoupled(ncells=6):
    """`ncells` sites with NO bonds at all -- so the exact single-site result simply
    multiplies, and every deviation is the sampler's."""
    cfg = {"crystal_structure": {"lattice_vectors": [[6., 0, 0], [0, 6, 0], [0, 0, 6]],
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": S_SPIN}]},
           "interactions": {"heisenberg": [],
                            "single_ion_anisotropy": [{"value": D_SIA,
                                                       "axis": [0, 0, 1],
                                                       "atoms": ["A"]}]},
           "parameters": {"H_mag": B_FIELD, "H_dir": [0, 0, 1]},
           "parameter_order": ["H_mag", "H_dir"],
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}
    return SUNModel.from_generic_model(
        GenericSpinModel(cfg), params=[B_FIELD, [0, 0, 1]],
        supercell=[[ncells, 0, 0], [0, 1, 0], [0, 0, 1]])


def _Z(c):
    """int_simplex exp(-sum_i c_i p_i) dp, closed form (distinct c_i)."""
    c = np.asarray(c, float)
    return sum(np.exp(-c[i]) / np.prod([c[j] - c[i] for j in range(len(c)) if j != i])
               for i in range(len(c)))


def _exact_energy(kT, n_sites):
    beta = 1.0 / kT
    h = 1e-5 * beta
    lnZ = lambda b: np.log(_Z(b * LEVELS))          # noqa: E731
    return -(lnZ(beta + h) - lnZ(beta - h)) / (2 * h) * n_sites


def _run(model, kT, sigma, adapt, n_sweeps=800, seed=3):
    Z = [np.asarray(z, complex).copy() for z in model.Z]
    _, info = sd.thermalize(model, Z, kT, n_sweeps, np.random.default_rng(seed),
                            sigma=sigma, adapt=adapt, on_unequilibrated="off",
                            return_info=True)
    half = len(info.energies) // 2
    return float(info.energies[half:].mean()), info


# --------------------------------------------------------------------------
def test_the_closed_form_partition_function_is_right():
    """Check the oracle before using it -- a wrong oracle certifies a wrong sampler."""
    from scipy.integrate import dblquad

    c = np.array([0.7, 0.0, 2.3])
    numeric = dblquad(lambda p2, p1: np.exp(-(c[0] * p1 + c[1] * p2
                                              + c[2] * (1 - p1 - p2))),
                      0, 1, lambda p1: 0, lambda p1: 1 - p1)[0]
    assert _Z(c) == pytest.approx(numeric, rel=1e-10)


def test_the_onsite_operator_is_the_one_the_oracle_assumes():
    """The closed form is a statement about THIS model's levels, so pin them: a
    change in the SIA/Zeeman convention must break this test rather than silently
    move both the sampler and the "exact" answer it is compared with."""
    model = _decoupled(2)
    total = np.zeros((3, 3), dtype=complex)
    for (site, A) in model.onsite:
        if site == 0:
            total = total + np.asarray(A)
    assert np.allclose(np.diag(total).real, LEVELS, atol=1e-12)
    assert np.allclose(total - np.diag(np.diag(total)), 0, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("kT", [0.05, 0.2, 1.0])
def test_the_adapted_sampler_reproduces_the_exact_energy(kT):
    """<E>(kT) from the sampler must be the closed-form value, at low kT too -- which
    is where the fixed step size failed."""
    model = _decoupled()
    exact = _exact_energy(kT, model.L)
    got, info = _run(model, kT, sigma=0.5, adapt=True, n_sweeps=1600)
    assert got == pytest.approx(exact, rel=0.04), f"kT={kT}, {info}"


@pytest.mark.slow
@pytest.mark.parametrize("kT", [0.05, 0.2])
def test_the_answer_no_longer_depends_on_where_sigma_started(kT):
    """THE property this item is about. Two starting step sizes 25x apart must give
    the same physics; `sigma` is a convergence knob, and an answer that moves with it
    is an unconverged answer wearing a plausible face."""
    model = _decoupled()
    lo, info_lo = _run(model, kT, sigma=0.02, adapt=True, n_sweeps=1600)
    hi, info_hi = _run(model, kT, sigma=0.5, adapt=True, n_sweeps=1600)
    assert lo == pytest.approx(hi, rel=0.04), f"{info_lo} vs {info_hi}"
    # ... because both were tuned to the same acceptance, from opposite directions
    assert 0.3 < info_lo.acceptance < 0.95
    assert info_lo.sigma == pytest.approx(info_hi.sigma, rel=0.5)


def test_the_fixed_step_sampler_is_measurably_wrong():
    """The defect, stated as a number, so the fix cannot be undone quietly.

    With `adapt=False` and the small step the old callers used, the chain accepts
    ~99 % of its proposals and explores almost nothing: at kT = 1 it reports
    <E> ~ -2.6 where the exact answer is -3.78, a 32 % error -- with no imaginary
    modes, no failed Cholesky and nothing else to notice it by.
    """
    model = _decoupled()
    exact = _exact_energy(1.0, model.L)
    stuck, info = _run(model, 1.0, sigma=0.02, adapt=False, n_sweeps=800)
    assert abs(stuck - exact) > 0.2 * abs(exact), f"{stuck} vs {exact}"
    assert info.acceptance > 0.95        # accepts everything, moves nowhere
    good, _ = _run(model, 1.0, sigma=0.02, adapt=True, n_sweeps=800)
    assert good == pytest.approx(exact, rel=0.05)


def test_the_equilibration_check_fires_on_the_stuck_run_and_not_the_good_one():
    """A diagnostic that fires on everything is as useless as one that fires on
    nothing, so both directions are asserted."""
    model = _decoupled()
    _stuck, bad = _run(model, 0.05, sigma=0.02, adapt=False, n_sweeps=800)
    _ok, good = _run(model, 0.05, sigma=0.02, adapt=True, n_sweeps=800)
    assert not bad.equilibrated and bad.drift > 1.0
    assert good.equilibrated and good.drift <= 1.0


def test_on_unequilibrated_error_refuses_and_off_is_silent(caplog):
    model = _decoupled()
    Z = [np.asarray(z, complex).copy() for z in model.Z]
    with pytest.raises(RuntimeError, match="NOT equilibrated"):
        sd.thermalize(model, Z, 0.05, 800, np.random.default_rng(3), sigma=0.02,
                      adapt=False, on_unequilibrated="error")
    caplog.clear()
    with caplog.at_level("WARNING"):
        Z = [np.asarray(z, complex).copy() for z in model.Z]
        sd.thermalize(model, Z, 0.05, 800, np.random.default_rng(3), sigma=0.02,
                      adapt=False, on_unequilibrated="off")
    assert caplog.text == ""
    caplog.clear()
    with caplog.at_level("WARNING"):
        Z = [np.asarray(z, complex).copy() for z in model.Z]
        sd.thermalize(model, Z, 0.05, 800, np.random.default_rng(3), sigma=0.02,
                      adapt=False, on_unequilibrated="warn")
    assert "NOT equilibrated" in caplog.text


def test_thermalize_still_returns_Z_by_default():
    """The signature grew; it must not have changed for existing callers."""
    model = _decoupled(2)
    Z = [np.asarray(z, complex).copy() for z in model.Z]
    out = sd.thermalize(model, Z, 0.5, 20, np.random.default_rng(0))
    assert out is Z
    assert [float(np.linalg.norm(z)) for z in Z] == pytest.approx([1.0] * model.L,
                                                                  abs=1e-12)


def test_the_sampler_knobs_reach_sampled_correlations():
    """`adapt_sigma` / `target_acceptance` / `on_unequilibrated` are config keys
    (`sun_sampled_correlations:`), so the path from the config to the sampler needs a
    test of its own -- see `tests/test_config_key_coverage.py` for why a key with a
    default and no test is the failure mode being guarded against here."""
    model = _decoupled(4)
    B = 2 * np.pi * np.linalg.inv(np.array([[6., 0, 0], [0, 6, 0], [0, 0, 6]])).T
    qs = np.array([[0.25, 0, 0]]) @ B
    seen = {}
    real = sd.thermalize

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    sd.thermalize = spy
    try:
        sd.sampled_correlations(model, qs, kT=0.5, dt=0.05, n_steps=32, n_traj=1,
                                therm_sweeps=20, seed=0, adapt_sigma=False,
                                target_acceptance=0.31, on_unequilibrated="off")
    finally:
        sd.thermalize = real
    assert seen["adapt"] is False
    assert seen["target_acceptance"] == 0.31
    assert seen["on_unequilibrated"] == "off"
