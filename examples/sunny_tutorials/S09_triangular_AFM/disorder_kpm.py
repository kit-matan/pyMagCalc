"""S09 -- exchange disorder broadening via KPM (Sunny tutorial 09).

    python examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py             # L=12
    python examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py --L 30      # Sunny's size
    python examples/sunny_tutorials/S09_triangular_AFM/disorder_kpm.py --sigma 0.333

The Hamiltonian is read from `config_supercell.yaml` (the single source of truth);
everything below is the tutorial's protocol: enlarge the cell, randomize every
nearest-neighbour exchange, re-relax, and compute S(q,w) with the kernel polynomial
method, which needs no eigensolve and so stays linear in the system size.

WHY A SCRIPT AND NOT A `tasks:` ENTRY. Bond disorder is a per-bond property of a
large real-space cell, applied AFTER the clean ground state is found and followed by
a re-relaxation. There is no config surface for that sequence, and inventing one
would hide the step that actually matters (the re-relaxation).

READ THIS BEFORE TRUSTING A DISORDERED SPECTRUM. KPM never diagonalizes, so --
unlike every other path in this engine -- it CANNOT notice that it is expanding
about a non-minimum: it will return a smooth, plausible, wrong S(q,w). This script
therefore checks stability explicitly (`max |Im w|` over the path, and the sharper
`min eig H2 >= 0`) and refuses by default when the check fails.

That is not a hypothetical. At Sunny's own disorder strength, sigma = 1/3, the
relaxed 120-degree state IS unstable on this model -- see the README for the
measured sigma sweep. The default here is therefore sigma = 0.1, which is stable,
and `--sigma 0.333 --force` reproduces the tutorial's setting with the warning.
"""
import argparse
import copy
import os
import time

import numpy as np
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.kpm import kpm_sqw
from magcalc.sun.lswt import SUNModel, apply_bond_disorder
from magcalc.sun.operators import coherent_from_direction

HERE = os.path.dirname(os.path.abspath(__file__))
K120 = np.array([1 / 3, 1 / 3, 0.0])          # the ordering wavevector (K point)
E_EXACT = -0.375                              # (1/2) * 6 * J * S^2 * cos(120 deg)


def _lattice(cs):
    p = cs["lattice_parameters"]
    a, b, c = p["a"], p["b"], p["c"]
    al, be, ga = (np.radians(p[k]) for k in ("alpha", "beta", "gamma"))
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, 0, 0],
                     [b * np.cos(ga), b * np.sin(ga), 0],
                     [cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))]], float)


def build(L):
    """The shipped Hamiltonian on an L x L triangular supercell, in the exact
    120-degree state. L must be a multiple of 3 for that state to fit."""
    if L % 3:
        raise SystemExit(f"--L must be a multiple of 3 (the 120-degree state has a "
                         f"3-sublattice cell); got {L}.")
    cfg = yaml.safe_load(open(os.path.join(HERE, "config_supercell.yaml")))
    cfg["crystal_structure"].pop("magnetic_supercell", None)   # we supply our own
    lat = _lattice(cfg["crystal_structure"])
    m = GenericSpinModel(copy.deepcopy(cfg))
    pv = []
    for k in cfg["parameter_order"]:
        v = cfg["parameters"][k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    # `config_supercell.yaml` carries no `magnetic_structure` (it finds the ground
    # state with tasks.minimization), so seed with anything and overwrite below.
    n_sites = L * L * len(m.atom_pos())
    mdl = SUNModel.from_generic_model(
        m, params=pv, directions=[[0.0, 0.0, 1.0]] * n_sites,
        supercell=[[L, 0, 0], [0, L, 0], [0, 0, 1]])

    # Impose the 120-degree state analytically rather than searching for it: on a
    # cell this large the CP^(N-1) search would land in one of many near-degenerate
    # domain configurations, and the point here is a controlled starting state.
    frac = np.asarray(mdl.pos, float) @ np.linalg.inv(lat)
    ang = 2 * np.pi * (frac @ K120)
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    mdl.Z = [coherent_from_direction(s, d) for s, d in zip(mdl.S, dirs)]
    mdl._prepare()
    return mdl, lat


def q_path(n_per_seg=40):
    """Gamma - K - M - Gamma, Sunny's path, in chemical RLU."""
    pts = [np.zeros(3), K120, np.array([0.5, 0.0, 0.0]), np.zeros(3)]
    qs, ticks = [], [0]
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            qs.append(a + t * (b - a))
        ticks.append(len(qs))
    qs.append(pts[-1])
    return np.array(qs), ticks


def stability(mdl, qs_cart):
    """(max |Im w|, min eig H2) over the path. H2 >= 0 at every q is the exact
    criterion for the reference state to be a classical minimum; the imaginary
    parts are what that failure looks like downstream."""
    im, lo = 0.0, np.inf
    for q in qs_cart:
        H = np.asarray(mdl.hamiltonian(q), complex)
        D = H.shape[0] // 2
        g = np.concatenate([np.ones(D), -np.ones(D)])
        H2 = g[:, None] * H                       # g @ (g H2) = H2
        lo = min(lo, float(np.linalg.eigvalsh(0.5 * (H2 + H2.conj().T)).min()))
        im = max(im, float(np.max(np.abs(np.imag(np.linalg.eigvals(H))))))
    return im, lo


def sqw(mdl, qs_cart, energies, fwhm, tol):
    out = np.zeros((len(energies), len(qs_cart)))
    for iq, q in enumerate(qs_cart):
        out[:, iq] = kpm_sqw(mdl, q, energies, fwhm, tol=tol,
                             cross_section="perp").intensities
    return out


W_MAX = 3 * 1.0 * 0.5 * np.sqrt(9 / 8)        # 1.59099 meV, the EXACT clean bandwidth


def generic_q(qs_rlu, cut=0.06):
    """Mask out q near Gamma and near K. Both carry a Goldstone mode, where S(q,w)
    diverges as w -> 0; those few columns dominate any path-averaged number and hide
    what happens everywhere else. Excluding them is not cherry-picking -- the
    divergence is a property of the CLEAN spectrum and is untouched by disorder."""
    q = np.asarray(qs_rlu, float)
    d0 = np.linalg.norm(q, axis=1)
    dk = np.min([np.linalg.norm(q - s * K120, axis=1) for s in (1, -1, 2)], axis=0)
    return (d0 > cut) & (dk > cut)


def metrics(smap, energies, fwhm):
    """Three ways to say 'broader', all monotone in the disorder strength:

      width  -- sqrt(<E^2> - <E>^2) per q, averaged;
      peak   -- the tallest feature, which a broadening must lower;
      above  -- the fraction of the weight above the EXACT clean bandwidth
                w_max = 3 J S sqrt(9/8). The clean spectrum has none there by
                construction, so this one is pinned to analysis rather than to a
                comparison, and it is the cleanest statement that the discrete
                bands have spread into a continuum.
    """
    w = np.maximum(smap, 0.0)[:, MASK]
    n = np.where(w.sum(axis=0) > 0, w.sum(axis=0), 1.0)
    e1 = (energies[:, None] * w).sum(axis=0) / n
    e2 = (energies[:, None] ** 2 * w).sum(axis=0) / n
    width = float(np.mean(np.sqrt(np.maximum(e2 - e1 ** 2, 0.0))))
    above = float(w[energies > W_MAX + 2 * fwhm].sum() / w.sum())
    return width, float(w.max()), above


def plot(energies, clean, dis, ticks, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.percentile(np.concatenate([clean, dis]), 99.5))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ext = [0, clean.shape[1], energies[0], energies[-1]]
    for a, m, t in ((ax[0], clean, "clean"),
                    (ax[1], dis, f"sigma = {args.sigma:g} ({args.seeds} seeds)")):
        a.imshow(m, origin="lower", aspect="auto", extent=ext, vmin=0, vmax=vmax,
                 cmap="magma")
        a.set_xticks(ticks)
        a.set_xticklabels(["$\\Gamma$", "K", "M", "$\\Gamma$"])
        a.set_ylabel("E (meV)")
        a.set_title(t)
    iq = ticks[1] + (ticks[2] - ticks[1]) // 2       # a cut between K and M
    ax[2].plot(energies, clean[:, iq], label="clean")
    ax[2].plot(energies, dis[:, iq], label=f"sigma = {args.sigma:g}")
    ax[2].set_xlabel("E (meV)")
    ax[2].set_ylabel("S(q, E)")
    ax[2].set_title("cut midway between K and M")
    ax[2].legend()
    fig.suptitle(f"S09: exchange disorder on the 120-degree triangular AFM, "
                 f"KPM, L = {args.L} ({args.L ** 2} sites)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=12, help="supercell edge in chemical "
                                                     "cells (multiple of 3)")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="relative exchange disorder; Sunny's tutorial uses 1/3")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--fwhm", type=float, default=0.2)
    p.add_argument("--tol", type=float, default=0.02, help="KPM tolerance")
    p.add_argument("--points", type=int, default=40, help="q per path segment")
    p.add_argument("--force", action="store_true",
                   help="continue even if the disordered state is not a minimum")
    p.add_argument("--out", default=os.path.join(HERE, "S09_disorder_kpm.png"))
    args = p.parse_args()

    qs_rlu, ticks = q_path(args.points)
    energies = np.arange(0.0, 3.0001, 0.02)
    global MASK
    MASK = generic_q(qs_rlu)

    t0 = time.time()
    mdl, lat = build(args.L)
    B = 2 * np.pi * np.linalg.inv(lat).T
    qs = qs_rlu @ B
    e0 = mdl.energy_per_site()
    print(f"L = {args.L} ({mdl.L} sites, {len(mdl.bonds)} directed bonds), "
          f"built in {time.time() - t0:.1f} s")
    print(f"clean 120-degree state: E/site = {e0:+.8f} meV "
          f"(exact {E_EXACT:+.6f}, dev {abs(e0 - E_EXACT):.1e})")
    if abs(e0 - E_EXACT) > 1e-8:
        raise SystemExit("the clean reference state is NOT the 120-degree ground "
                         "state -- stop here, every spectrum below would be about "
                         "the wrong state.")

    t0 = time.time()
    clean = sqw(mdl, qs, energies, args.fwhm, args.tol)
    m_clean = metrics(clean, energies, args.fwhm)
    print(f"clean KPM: {len(qs)} q x {len(energies)} E in {time.time() - t0:.1f} s; "
          f"width {m_clean[0]:.4f} meV, peak {m_clean[1]:.3f}, "
          f"weight above the exact bandwidth {m_clean[2]:.4f}")

    acc = np.zeros_like(clean)
    rows = []
    for seed in range(args.seeds):
        mdl, _ = build(args.L)
        apply_bond_disorder(mdl, args.sigma, seed=seed)
        t0 = time.time()
        e = mdl.minimize_energy(n_restarts=1, max_iter=3000, tol=1e-15) / mdl.L
        im, lo = stability(mdl, qs[::5])
        t_rel = time.time() - t0
        smap = sqw(mdl, qs, energies, args.fwhm, args.tol)
        acc += smap
        mt = metrics(smap, energies, args.fwhm)
        rows.append((seed, e, im, lo) + mt)
        print(f"  seed {seed}: E/site = {e:+.8f}  max|Im w| = {im:.2e}  "
              f"min eig H2 = {lo:+.2e}  width {mt[0]:.4f}  peak {mt[1]:.3f}  "
              f"above {mt[2]:.4f}  ({t_rel:.1f} s relax)")
        if lo < -1e-6 and not args.force:
            raise SystemExit(
                f"\nThe relaxed disordered state is NOT a classical minimum "
                f"(min eig H2 = {lo:.2e}, max |Im w| = {im:.2e} on a "
                f"{3 * 0.5 * np.sqrt(9 / 8):.3f} meV band).\n"
                f"LSWT about a non-minimum is meaningless, and KPM cannot detect it "
                f"for you -- it never diagonalizes.\n"
                f"  * lower --sigma (0.1 is stable on this model; 1/3 is not), or\n"
                f"  * pass --force to compute it anyway and read the result as the "
                f"tutorial's illustration rather than as physics.")
    dis = acc / args.seeds
    m_dis = metrics(dis, energies, args.fwhm)

    print(f"\n                        clean      sigma = {args.sigma:g}")
    print(f"  width (meV)        {m_clean[0]:9.4f} {m_dis[0]:14.4f}   "
          f"({100 * (m_dis[0] / m_clean[0] - 1):+.1f} %)")
    print(f"  peak               {m_clean[1]:9.3f} {m_dis[1]:14.3f}   "
          f"({100 * (m_dis[1] / m_clean[1] - 1):+.1f} %)")
    print(f"  weight above w_max {m_clean[2]:9.4f} {m_dis[2]:14.4f}")
    print(f"seed-to-seed spread of the width: "
          f"{np.std([r[4] for r in rows]) / np.mean([r[4] for r in rows]):.1%}")
    plot(energies, clean, dis, ticks, args, args.out)


if __name__ == "__main__":
    main()
