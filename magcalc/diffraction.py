"""Elastic magnetic neutron diffraction: the magnetic structure factor.

This is the *static* counterpart of the S(Q, w) layer -- what a diffractometer
measures from an ordered magnetic structure, as opposed to what a spectrometer
measures from its excitations. It exists so that a magnetic structure carried by
this package (from an mCIF, from a refinement, or from a minimisation) can have
its Bragg intensities computed **independently of the refinement program that
produced it**.

    F(Q)   = sum_i  f_i(Q) m_i exp(i Q.r_i)             (vector, Bohr magnetons)
    F_perp = F - Qhat (Qhat . F)                        (only this scatters)
    I(Q)  proportional to |F_perp(Q)|^2

`f_i` is the dipole-approximation magnetic form factor already tabulated in
`magcalc.form_factors`; `r_i` are Cartesian positions in Angstrom and Q is
Cartesian in 1/Angstrom, so `Q.r` is the usual 2*pi*(hx+ky+lz) when Q is built
from the reciprocal lattice by `q_cartesian`.

The absolute prefactor (gamma r_0 / 2)^2 = 0.07265 barn/sr per mu_B^2, together
with Lorentz factors, multiplicity, absorption and the scale factor, is what a
Rietveld program fits; none of it changes the RATIO between reflections of one
pattern, so this module returns |F_perp|^2 in mu_B^2 and the comparisons in
`intensity_ratio_report` are ratio-based on purpose.

Why the coherent sum is the whole point
---------------------------------------
`magnetic_intensity` sums over **every magnetic site of the structure** before
squaring. A refinement that enters two crystallographically inequivalent
sublattices as two separate magnetic *phases* gets

    I_incoherent = |F_1|^2 + |F_2|^2      instead of      |F_1 + F_2|^2

and silently loses the cross term 2 Re(F_1 . F_2*) -- which is exactly the
interference that distinguishes one candidate magnetic structure from another.
That error is not hypothetical: it produced a published magnetic structure that
had to be withdrawn (Cs2Cu3SnF12; see DISCOVERY_WORKFLOW_PLAN.md section 9). It
converges, it reports a respectable chi^2, and nothing inside the refinement
notices. `incoherent_intensity` computes the wrong answer deliberately, so that
`cross_term_fraction` can measure how far apart the two are on a given
reflection, and a pipeline can refuse to accept a model where they differ.
"""
from __future__ import annotations

import numpy as np

from magcalc.form_factors import get_form_factor

__all__ = [
    "reciprocal_lattice", "q_cartesian", "structure_factor", "perpendicular",
    "magnetic_intensity", "incoherent_intensity", "cross_term_fraction",
    "intensity_ratio_report",
]


def reciprocal_lattice(lattice_vectors) -> np.ndarray:
    """Reciprocal lattice vectors as ROWS, 1/Angstrom, with the 2*pi.

    `lattice_vectors` are the direct lattice vectors as rows (the convention
    used by `crystal_structure.lattice_vectors` and by `mcif.read_mcif`), so
    B = 2*pi * inv(A).T and B @ A.T = 2*pi I.
    """
    A = np.asarray(lattice_vectors, dtype=float)
    if A.shape != (3, 3):
        raise ValueError(f"lattice_vectors must be 3x3, got {A.shape}")
    return 2.0 * np.pi * np.linalg.inv(A).T


def q_cartesian(hkl, lattice_vectors) -> np.ndarray:
    """Cartesian scattering vector(s) for Miller indices, 1/Angstrom."""
    B = reciprocal_lattice(lattice_vectors)
    hkl = np.atleast_2d(np.asarray(hkl, dtype=float))
    return hkl @ B


def _form_factors(ions, Q_mag, gs) -> np.ndarray:
    """f_i(|Q|) for each site; 1.0 where the ion is unknown or unset."""
    if ions is None:
        return np.ones(len(gs))
    out = []
    for ion, g in zip(ions, gs):
        out.append(1.0 if ion in (None, "") else float(get_form_factor(ion, Q_mag, g=g)))
    return np.asarray(out, dtype=float)


def structure_factor(Q, positions, moments, ions=None, g=2.0) -> np.ndarray:
    """The (complex, vector) magnetic structure factor F(Q), in mu_B.

    Args:
        Q: Cartesian scattering vector, 1/Angstrom, shape (3,).
        positions: Cartesian site positions in Angstrom, shape (N, 3).
        moments: Cartesian magnetic moments in mu_B, shape (N, 3).
        ions: per-site form-factor labels ('Cu2+', ...), or None for f = 1.
            A None entry means that site alone gets f = 1.
        g: scalar or per-site g factor, used only by the <j2> correction.

    Returns:
        Complex array of shape (3,) -- F is a VECTOR because the moment
        direction matters; `perpendicular` then projects out the component
        along Q, which is the only part that scatters.
    """
    Q = np.asarray(Q, dtype=float).reshape(3)
    r = np.asarray(positions, dtype=float).reshape(-1, 3)
    m = np.asarray(moments, dtype=float).reshape(-1, 3)
    if len(r) != len(m):
        raise ValueError(f"{len(r)} positions but {len(m)} moments")
    gs = np.full(len(r), float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)
    f = _form_factors(ions, float(np.linalg.norm(Q)), gs)
    phase = np.exp(1j * (r @ Q))                       # (N,)
    return (f[:, None] * m * phase[:, None]).sum(axis=0)


def perpendicular(F, Q) -> np.ndarray:
    """The component of F perpendicular to Q (the magnetic interaction vector).

    Neutrons see only this: F_perp = F - Qhat (Qhat . F). At Q = 0 the direction
    is undefined and the whole vector is returned unchanged.
    """
    F = np.asarray(F)
    Q = np.asarray(Q, dtype=float).reshape(3)
    n = np.linalg.norm(Q)
    if n < 1e-12:
        return F
    qhat = Q / n
    return F - qhat * (qhat @ F)


def magnetic_intensity(Q, positions, moments, ions=None, g=2.0) -> float:
    """|F_perp(Q)|^2 in mu_B^2 -- the COHERENT sum over every site.

    This is the quantity a single-phase Rietveld refinement computes, and the
    reference against which a multi-phase magnetic refinement should be checked.
    """
    F = structure_factor(Q, positions, moments, ions=ions, g=g)
    Fp = perpendicular(F, Q)
    return float(np.real(np.vdot(Fp, Fp)))


def incoherent_intensity(Q, positions, moments, groups, ions=None, g=2.0) -> float:
    """sum_p |F_perp^(p)(Q)|^2 -- the WRONG answer, computed on purpose.

    `groups` labels each site with the magnetic phase it was entered as (any
    hashable; e.g. ['Cu1', 'Cu1', 'Cu2', ...]). Squaring each phase separately
    and adding is what a multi-phase refinement does, and it drops the
    interference between phases. Compare against `magnetic_intensity` -- see
    `cross_term_fraction`.
    """
    groups = list(groups)
    if len(groups) != len(np.asarray(positions).reshape(-1, 3)):
        raise ValueError("groups must have one label per site")
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    moments = np.asarray(moments, dtype=float).reshape(-1, 3)
    gs = np.full(len(positions), float(g)) if np.isscalar(g) else np.asarray(g, dtype=float)
    total = 0.0
    for label in dict.fromkeys(groups):                # stable unique
        sel = [i for i, gl in enumerate(groups) if gl == label]
        sub_ions = None if ions is None else [ions[i] for i in sel]
        total += magnetic_intensity(Q, positions[sel], moments[sel],
                                    ions=sub_ions, g=gs[sel])
    return float(total)


def cross_term_fraction(Q, positions, moments, groups, ions=None, g=2.0) -> float:
    """(I_coherent - I_incoherent) / max(I_coherent, I_incoherent), in [-1, 1].

    How much the coherent and multi-phase treatments disagree on one reflection.
    Zero where the phases happen not to interfere -- so a SMALL value proves
    nothing about a model, and only a large one discriminates. A candidate
    ranking produced by a multi-phase refinement must be checked on the
    reflections where this is large.

    The denominator is the LARGER of the two, not the coherent one, because the
    most damaging case is the one where the coherent intensity is small: at a
    systematic absence of the true structure factor (I_coherent = 0) a two-phase
    refinement still predicts |F_1|^2 + |F_2|^2 > 0, i.e. a peak that should not
    be there at all. That is maximal disagreement and returns -1; dividing by
    I_coherent would return -inf and lose the reflection to a numerical guard,
    which is precisely the reflection worth looking at.
    """
    coh = magnetic_intensity(Q, positions, moments, ions=ions, g=g)
    inc = incoherent_intensity(Q, positions, moments, groups, ions=ions, g=g)
    denom = max(coh, inc)
    if denom < 1e-30:                      # nothing scatters either way
        return 0.0
    return float((coh - inc) / denom)


def intensity_ratio_report(hkls, lattice_vectors, positions, moments,
                           ions=None, g=2.0, groups=None, normalize=True):
    """Per-reflection intensities, ready to compare against a refinement.

    Returns a list of dicts with `hkl`, `d` (Angstrom), `Q` (1/Angstrom),
    `I` and -- when `groups` is given -- `I_incoherent` and `cross_fraction`.

    With `normalize` (the default) the intensities are scaled so the strongest
    reflection is 1. That is deliberate: the absolute scale of a refinement
    folds in the scale factor, the Lorentz factor and the sample, none of which
    this module models, so only ratios are comparable, and a comparison that
    pretends otherwise would fail for the wrong reason.
    """
    hkls = np.atleast_2d(np.asarray(hkls, dtype=float))
    Qs = q_cartesian(hkls, lattice_vectors)
    rows = []
    for hkl, Q in zip(hkls, Qs):
        qn = float(np.linalg.norm(Q))
        row = {
            "hkl": [int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in hkl],
            "Q": qn,
            "d": (2.0 * np.pi / qn) if qn > 1e-12 else float("inf"),
            "I": magnetic_intensity(Q, positions, moments, ions=ions, g=g),
        }
        if groups is not None:
            row["I_incoherent"] = incoherent_intensity(
                Q, positions, moments, groups, ions=ions, g=g)
            row["cross_fraction"] = cross_term_fraction(
                Q, positions, moments, groups, ions=ions, g=g)
        rows.append(row)
    if normalize and rows:
        peak = max(r["I"] for r in rows)
        if peak > 1e-30:
            for r in rows:
                r["I"] /= peak
                if "I_incoherent" in r:
                    r["I_incoherent"] /= peak
    return rows
