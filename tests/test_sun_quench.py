"""The DISSIPATIVE CP^(N-1) flow and the Berg-Luescher charge (Sunny tutorial 06).

`sun/dynamics.py` grew a second flow when Gap 4 #26 closed: alongside the
energy-conserving `_deriv` it has `damped_deriv` / `damped_step` / `quench`, plus
`topological_charge` and `triangulate_lattice`. **None of them had a test.** The
S06 notes recorded that the quench had been "validated by dE/dt = -2 lambda Var(h)
to 5e-6", but that was a one-off measurement at a terminal, not a pinned check, so
nothing in the suite would have noticed the flow changing sign or the charge losing
its quantization. This file is that oracle, written before the derivative was
rewritten for speed.

Every check below is an EXACT IDENTITY of the equations of motion, not a recorded
number:

  * lambda = 0 must reproduce the conservative flow BIT FOR BIT -- it is the same
    expression with a term switched off;
  * d|Z_i|^2/dt = 0 identically, because <h> is real for Hermitian h. The flow
    preserves the CP^(N-1) constraint by itself; `_renormalize` is a numerical
    tidy-up, not the thing that keeps the state legal;
  * dE/dt = -2 lambda sum_i Var_i(h). This is the load-bearing one. It ties the
    DERIVATIVE to the ENERGY the model reports, through two independently written
    functions, and it fixes the SIGN -- the failure mode that actually happened in
    the dipole engine (Gap 4 #18 relaxed spins AWAY from the minimum and produced a
    right-magnitude, wrong-sign magnetization);
  * the fixed point is an EIGENVECTOR of the local field, and specifically the
    LOWEST one. A sign slip would land on the highest, which is still a fixed point
    and still looks converged;
  * the Berg-Luescher charge of a texture that wraps the sphere once is 1, and of a
    uniform texture is 0 EXACTLY. Quantization is what makes the skyrmion count an
    assertion rather than something eyeballed off a colour map.

The last test is not about the dynamics at all: it checks that
`triangulate_lattice`'s assumed site layout is the one `SUNModel._replicate`
actually produces. A mismatch there would scramble the texture while leaving every
energy in this file correct.
"""
import copy
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun import dynamics as sd
from magcalc.sun.lswt import SUNModel

HERE = os.path.dirname(__file__)
S06 = os.path.join(HERE, "..", "examples", "sunny_tutorials", "S06_CP2_skyrmions",
                   "config.yaml")


def _s06_model(L=4):
    """The shipped S06 Hamiltonian on an L x L triangular supercell."""
    cfg = yaml.safe_load(open(S06))
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    pv = []
    for k in cfg["parameter_order"]:
        v = cfg["parameters"][k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    return SUNModel.from_generic_model(m, params=pv,
                                       supercell=[[L, 0, 0], [0, L, 0], [0, 0, 1]])


def _random_state(mdl, seed):
    rng = np.random.default_rng(seed)
    Z = []
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        Z.append(v / np.linalg.norm(v))
    return Z


def _variance_sum(mdl, Z):
    """sum_i (<h_i^2> - <h_i>^2) for the current configuration."""
    tot = 0.0
    for i, h in enumerate(sd.local_hamiltonians(mdl, Z)):
        hZ = h @ Z[i]
        tot += float(np.real(hZ.conj() @ hZ)) - float(np.real(Z[i].conj() @ hZ)) ** 2
    return tot


def _reference_deriv(mdl, Z, lam=None):
    """The derivative written the slow, obvious way: form each h_i, apply it.

    Deliberately built from `SUNModel.local_field`, which is the DEFINITION of the
    mean field and is exercised independently by the CP^(N-1) ground-state search.
    """
    out = []
    for i, h in enumerate(sd.local_hamiltonians(mdl, Z)):
        hZ = h @ Z[i]
        if lam is None:
            out.append(-1j * hZ)
        else:
            exp_h = np.real(Z[i].conj() @ hZ)
            out.append(-1j * hZ - lam * (hZ - exp_h * Z[i]))
    return out


def _two_site_model(S=1.0, biquad=None, spins=None):
    """A small model with an anisotropy, optionally biquadratic or mixed spin."""
    lat = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
    S_A, S_B = (spins if spins else (S, S))
    nn = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
          (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": 0.8}
                            for p, o in nn],
             "single_ion_anisotropy": [{"value": -0.45, "axis": [0.3, 0, 1],
                                        "atoms": ["A", "B"]}]}
    if biquad is not None:
        inter["biquadratic"] = [{"pair": p, "rij_offset": o, "value": biquad}
                                for p, o in nn]
    cfg = {"crystal_structure": {"lattice_vectors": lat,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": S_A},
                                              {"label": "B", "pos": [0.5, 0, 0],
                                               "spin_S": S_B}]},
           "interactions": inter, "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)),
                                       params=[])


@pytest.mark.parametrize("mdl_kwargs,label", [
    ({}, "plain SU(3)"),
    ({"biquad": -0.3}, "biquadratic (12-wide operator set)"),
    ({"spins": (0.5, 1.0)}, "mixed spin (falls back to the per-site loop)"),
])
def test_the_fast_derivative_equals_the_local_field_definition(mdl_kwargs, label):
    """The vectorized inner loop is a change of LOOP ORDER, nothing else.

    `_h_times_z` never forms h_i and sums the bond list once instead of once per
    site, so it has to be checked against the definition it replaced. Three cases,
    because the shortcut has three ways to be wrong:

      * plain SU(3) -- the ordinary path;
      * BIQUADRATIC, where `n_ops` is 3 + 9 and the coupling acts on the products
        S^a S^b. Hardcoding the three dipoles here would minimize a different
        Hamiltonian than `hamiltonian()` diagonalizes -- the exact trap
        `local_field`'s own docstring warns about;
      * MIXED SPIN, where the per-site Hilbert spaces have different dimensions and
        cannot be stacked at all. That must take the fallback, so this case checks
        the fallback is still reachable and still correct.
    """
    mdl = _two_site_model(**mdl_kwargs)
    assert sd._fast(mdl)["ok"] == ("mixed" not in label)
    Z = _random_state(mdl, 17)
    for lam in (None, 0.23):
        got = sd._deriv(mdl, Z) if lam is None else sd.damped_deriv(mdl, Z, lam)
        want = _reference_deriv(mdl, Z, lam)
        err = max(np.abs(a - b).max() for a, b in zip(got, want))
        scale = max(np.abs(b).max() for b in want)
        assert err <= 1e-12 * scale, f"{label}, lam={lam}: {err:.3e}"


def test_the_bond_cache_notices_a_changed_hamiltonian():
    """A cache keyed on the wrong thing is a silently stale Hamiltonian.

    The stacked bond arrays are built once and hung on the model, so removing or
    replacing bonds must invalidate them. Checked by REMOVING every bond: the
    derivative then has to be the pure on-site one, not the bonded answer it had
    just computed.
    """
    mdl = _two_site_model()
    Z = _random_state(mdl, 4)
    sd._deriv(mdl, Z)                          # populate the cache
    mdl.bonds = []
    assert max(np.abs(a - b).max()
               for a, b in zip(sd._deriv(mdl, Z), _reference_deriv(mdl, Z))) < 1e-12


def test_zero_damping_is_exactly_the_conservative_flow():
    """`damped_deriv(lambda=0)` and `_deriv` are the same expression."""
    mdl = _s06_model(L=3)
    Z = _random_state(mdl, 7)
    a = sd._deriv(mdl, Z)
    b = sd.damped_deriv(mdl, Z, 0.0)
    assert max(np.abs(x - y).max() for x, y in zip(a, b)) == 0.0


def test_the_damped_flow_preserves_the_norm_by_itself():
    """d|Z_i|^2/dt = 2 Re[-i<h> - lambda(<h> - <h>)] = 0, for EVERY site.

    Checked on the raw derivative, before any renormalization: if this needed
    `_renormalize` to hold, the flow would not be a flow on CP^(N-1) and the
    'energy' being minimized would be that of a state of drifting length.
    """
    mdl = _s06_model(L=3)
    Z = _random_state(mdl, 21)
    dZ = sd.damped_deriv(mdl, Z, 0.37)
    for i in range(mdl.L):
        assert abs(float(np.real(Z[i].conj() @ dZ[i]))) < 1e-14


@pytest.mark.parametrize("lam", [0.05, 0.4])
def test_energy_decay_rate_is_minus_two_lambda_times_the_variance(lam):
    """dE/dt = -2 lambda sum_i Var_i(h) -- the identity that fixes the SIGN.

    The left side is a central difference of `SUNModel._energy_of` along the flow;
    the right side is built from `local_field`. Two separately written functions,
    so agreement is a statement about the physics rather than about one expression
    being copied twice. It also proves the flow DESCENDS: Var >= 0, so dE/dt <= 0.
    """
    mdl = _s06_model(L=3)
    Z = _random_state(mdl, 3)
    dt = 1e-4
    fwd = sd.damped_step(mdl, Z, dt, lam)
    bwd = sd.damped_step(mdl, Z, -dt, lam)
    numeric = (sd.energy(mdl, fwd) - sd.energy(mdl, bwd)) / (2 * dt)
    exact = -2.0 * lam * _variance_sum(mdl, Z)
    assert exact <= 0.0
    assert numeric == pytest.approx(exact, rel=1e-6)


def test_the_quench_descends_to_the_LOWEST_local_eigenvector():
    """The fixed point is <h_i>-self-consistency, and the sign picks which one.

    A single site in a field plus an easy-plane anisotropy has no bonds, so the
    local field is a FIXED matrix and the exact answer is its lowest eigenvector --
    no self-consistency, no ambiguity. Damping with the wrong sign converges just as
    happily onto the HIGHEST eigenvector, which is why 'the quench converged' is not
    evidence of anything on its own.
    """
    mdl = _s06_model(L=1)
    mdl.bonds = []                       # isolate one site: h is then constant
    h = sd.local_hamiltonians(mdl, [np.array([1, 0, 0], dtype=complex)])[0]
    w, v = np.linalg.eigh(h)

    Z0 = _random_state(mdl, 5)
    Zf, _ = sd.quench(mdl, Z0, dt=0.02, n_steps=4000, damping=0.1)
    overlap = abs(complex(v[:, 0].conj() @ Zf[0]))
    assert overlap == pytest.approx(1.0, abs=1e-6), (
        f"converged onto eigenvalue {float(np.real(Zf[0].conj() @ h @ Zf[0]))}, "
        f"spectrum {w}")


def test_the_quench_energy_never_rises():
    """Monotone descent on the real (bonded) model, where h_i is not constant."""
    mdl = _s06_model(L=4)
    Z = _random_state(mdl, 9)
    energies = [sd.energy(mdl, Z)]
    for _ in range(60):
        Z = sd.damped_step(mdl, Z, 0.01, 0.05)
        energies.append(sd.energy(mdl, Z))
    e = np.array(energies)
    assert np.all(np.diff(e) <= 1e-9), f"energy rose: {np.diff(e).max()}"
    assert e[-1] < e[0] - 1e-6


# --------------------------------------------------------------- the charge
def _skyrmion_texture(n1, n2, a1, a2, width):
    """A texture that wraps the sphere EXACTLY once, on the torus.

    theta(r) = pi exp(-r^2 / width^2), phi = atan2(y, x): as r runs 0 -> inf the
    polar angle sweeps pi -> 0 once while the azimuth winds once, i.e. degree 1.
    The Gaussian profile makes the texture UNIFORM (+z) at the cell boundary to
    ~1e-4, so it is smooth on the torus and the Berg-Luescher sum is entitled to be
    quantized.
    """
    out = np.zeros((n1 * n2, 3))
    for u in range(n1):
        for v in range(n2):
            # measure from the cell centre, folded to the nearest image
            du = (u + n1 // 2) % n1 - n1 // 2
            dv = (v + n2 // 2) % n2 - n2 // 2
            r = du * np.asarray(a1) + dv * np.asarray(a2)
            rad = float(np.hypot(r[0], r[1]))
            th = np.pi * np.exp(-(rad / width) ** 2)
            ph = np.arctan2(r[1], r[0])
            out[u * n2 + v] = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                               np.cos(th)]
    return out


def test_topological_charge_is_quantized():
    """|Q| = 1 for a single wrap of the sphere, 0 for a uniform texture.

    QUANTIZATION IS EXACT HERE, not approximate, and that is worth stating: the
    Berg-Luescher sum of signed spherical-triangle areas over a closed triangulated
    surface IS 4 pi times the degree of the map, a statement about the DISCRETE
    texture. There is no mesh-refinement error to allow for -- measured residual
    2e-16 -- so a loose tolerance here would only hide a real defect.

    The SIGN is -1, and it is not a convention artefact. For theta(r) sweeping pi
    at the core to 0 at the boundary with azimuth winding +1,

        Q = -(1/2) [cos theta(inf) - cos theta(0)] * w = -(1/2)(1 + 1)(+1) = -1,

    i.e. a texture whose core points DOWN and whose background points UP is degree
    -1. (This test was first written asserting +1; the code was right.)
    """
    n1 = n2 = 36
    a1 = np.array([1.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2])
    tris = sd.triangulate_lattice(np.zeros((n1 * n2, 3)), a1, a2, n1, n2)

    uniform = np.tile([0.0, 0.0, 1.0], (n1 * n2, 1))
    assert abs(sd.topological_charge(uniform, tris)) < 1e-12

    sky = _skyrmion_texture(n1, n2, a1, a2, width=6.0)
    assert sd.topological_charge(sky, tris) == pytest.approx(-1.0, abs=1e-9)
    # mirroring the texture reverses the winding: skyrmion <-> antiskyrmion
    anti = sky * np.array([1.0, -1.0, 1.0])
    assert sd.topological_charge(anti, tris) == pytest.approx(+1.0, abs=1e-9)


def test_the_SUN_charge_reduces_to_the_dipole_one_at_N2():
    """At N = 2 the CP^1 Berry phase of a triangle is half its solid angle.

    With the 2 pi / 4 pi normalizations that makes `sun_topological_charge` EQUAL
    to `topological_charge` for spin-1/2 coherent states -- an exact identity, and
    the oracle for the SU(N) charge: it pins the new quantity to the one already
    pinned above, with no new golden number. Away from N = 2 there is no such
    reduction, which is the whole point of having it.

    Checked on BOTH windings, so it is an identity about the charge and not a
    coincidence of two quantities that both happen to land on 1.
    """
    from magcalc.sun.operators import coherent_from_direction

    n1 = n2 = 24
    a1 = np.array([1.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2])
    tris = sd.triangulate_lattice(np.zeros((n1 * n2, 3)), a1, a2, n1, n2)
    sky = _skyrmion_texture(n1, n2, a1, a2, width=5.0)

    for tex, want in ((sky, -1.0), (sky * np.array([1.0, -1.0, 1.0]), +1.0)):
        Z = [coherent_from_direction(0.5, d) for d in tex]
        q_dip = sd.topological_charge(tex, tris)
        assert q_dip == pytest.approx(want, abs=1e-9)
        assert sd.sun_topological_charge(Z, tris) == pytest.approx(q_dip, abs=1e-9)


def test_the_dipole_charge_refuses_a_collapsed_moment():
    """A quadrupolar site has no dipole to take a solid angle of.

    `topological_charge` normalizes every spin, so a vanishing moment silently
    becomes an arbitrary unit vector and the sum comes back looking like a
    perfectly good quantized number. That is the exact situation in Sunny tutorial
    06, whose background is the |m=0> level -- so it must refuse rather than answer.
    """
    n1 = n2 = 8
    tris = sd.triangulate_lattice(np.zeros((n1 * n2, 3)), np.array([1.0, 0.0]),
                                  np.array([-0.5, np.sqrt(3) / 2]), n1, n2)
    spins = np.tile([0.0, 0.0, 1.0], (n1 * n2, 1))
    spins[5] = [1e-9, 0.0, -2e-9]                 # one collapsed dipole
    with pytest.raises(ValueError, match="collapsed dipole|sun_topological_charge"):
        sd.topological_charge(spins, tris)
    # the SU(N) charge is defined on the coherent states and has no such problem
    mdl = _s06_model(L=1)
    z = np.array([0.0, 1.0, 0.0], dtype=complex)   # the |m=0> quadrupolar state
    assert np.allclose(sd.dipole_field(mdl, [z]), 0.0, atol=1e-12)
    assert sd.sun_topological_charge([z] * (n1 * n2), tris) == pytest.approx(0.0)


def test_triangulate_lattice_agrees_with_the_supercell_site_order():
    """The plaquette indexing assumes a layout; check the model really has it.

    `triangulate_lattice` builds its triangles from index arithmetic alone
    (idx(u,v) = u*n2 + v), while the site order comes from `SUNModel._replicate`'s
    own cell enumeration. Nothing connected the two. If they disagreed, every energy
    in this file would still be right and the texture would be scrambled -- so the
    charge would be noise while looking like a computed number.
    """
    L = 4
    mdl = _s06_model(L=L)
    lat = np.array([[1.0, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 10.0]])
    assert mdl.L == L * L
    for u in range(L):
        for v in range(L):
            want = u * lat[0] + v * lat[1]
            assert np.allclose(mdl.pos[u * L + v], want, atol=1e-9), \
                f"site {u * L + v} at {mdl.pos[u * L + v]}, expected {want}"
