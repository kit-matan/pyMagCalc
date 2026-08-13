"""S06 -- CP^2 skyrmion liquid from a dynamical quench (Sunny tutorial 06).

    python examples/sunny_tutorials/S06_CP2_skyrmions/quench.py            # L = 40
    python examples/sunny_tutorials/S06_CP2_skyrmions/quench.py --L 16     # quick look

The tutorial's product is a non-equilibrium TEXTURE, not a spectrum, so this is a
script rather than a `tasks:` entry. The Hamiltonian is read from `config.yaml`
(the single source of truth); everything below is the quench protocol.

THE PROTOCOL IS THE POINT. Sunny randomizes the spins and integrates a damped
Langevin flow at kT = 0. Substituting a ground-state search or a Metropolis
equilibration would destroy the object of interest: skyrmions here are a METASTABLE
texture a quench leaves behind, not the ground state, which is uniformly polarized.
`config.yaml` run on its own computes exactly that uniform state -- the thing this
script must NOT reproduce.

WHY THE SYSTEM MUST BE THIS BIG. A skyrmion is several lattice constants across, so
a cell of 64 or 256 sites simply cannot hold a liquid of them; it relaxes to the
uniform state and looks like a physics failure rather than a size failure. Sunny
uses L = 40 (1600 sites), which was out of reach here until the CP^(N-1) derivative
was vectorized (it cost ~16 s/step, i.e. 55 hours for this run; it is now ~10 ms).

WHAT IS PLOTTED is the SU(3) Berry curvature per triangular plaquette, as in the
tutorial -- NOT the dipole solid angle. Most of the area is a quadrupolar
paramagnet with <S> ~ 0, where a dipole texture is undefined; `topological_charge`
refuses on exactly that ground.
"""
import argparse
import copy
import os
import time

import numpy as np
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun import dynamics as sd
from magcalc.sun.lswt import SUNModel

HERE = os.path.dirname(os.path.abspath(__file__))


def build(L):
    """The shipped Hamiltonian on an L x L triangular supercell."""
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    pv = []
    for k in cfg["parameter_order"]:
        v = cfg["parameters"][k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    return SUNModel.from_generic_model(
        m, params=pv, supercell=[[L, 0, 0], [0, L, 0], [0, 0, 1]])


def randomize(mdl, seed):
    """Uniform on CP^(N-1) -- Sunny's `randomize_spins!` for a :SUN system."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        out.append(v / np.linalg.norm(v))
    return out


def plot(frames, taus, L, a1, a2, tris, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # `triangulate_lattice` emits two triangles per cell, cells in (u, v) order.
    # Skip the ones that wrap the periodic boundary: they are legitimate for the
    # charge (the texture lives on a torus) and meaningless to draw.
    keep = [t for t in range(len(tris))
            if all(x + 1 < L for x in divmod(t // 2, L))]
    verts_all = [[((idx // L) * a1 + (idx % L) * a2)[:2] for idx in tri]
                 for tri in tris]

    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4.4))
    axes = np.atleast_1d(axes)
    for ax, tau, Z in zip(axes, taus, frames):
        dens = sd.berry_curvature(Z, tris)
        pc = PolyCollection([verts_all[t] for t in keep], array=dens[keep],
                            cmap="RdBu_r", edgecolors="none")
        pc.set_clim(-np.pi / 2, np.pi / 2)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"$\\tau$ = {tau}")
    fig.colorbar(pc, ax=axes.tolist(), shrink=0.8,
                 label="SU(3) Berry curvature per plaquette")
    fig.suptitle(f"S06: CP$^2$ skyrmion liquid after a quench ({L}x{L} sites)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=40, help="linear size (Sunny uses 40)")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--damping", type=float, default=0.05)
    p.add_argument("--taus", type=float, nargs="+", default=[4, 16, 256],
                   help="snapshot times, in hbar/|J1|")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "S06_skyrmions.png"))
    args = p.parse_args()

    t0 = time.time()
    mdl = build(args.L)
    print(f"model: {mdl.L} sites, {len(mdl.bonds)} bonds, N = {mdl.Ns[0]} "
          f"({time.time() - t0:.1f} s)")

    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2, 0.0])
    tris = sd.triangulate_lattice(mdl.pos, a1, a2, args.L, args.L)

    Z0 = randomize(mdl, args.seed)
    steps = [int(round(t / args.dt)) for t in args.taus]
    print(f"quench: dt = {args.dt}, damping = {args.damping}, "
          f"{max(steps)} steps to tau = {max(args.taus)}")

    t0 = time.time()
    _, snaps = sd.quench(mdl, Z0, args.dt, max(steps), damping=args.damping,
                         record_at=steps)
    print(f"  {time.time() - t0:.1f} s "
          f"({1e3 * (time.time() - t0) / max(steps):.2f} ms/step)")

    frames = [snaps[s] for s in steps]
    print(f"\n{'tau':>8} {'E/site':>12} {'<Sz>':>9} {'|<S>|':>9} {'Q_SU(3)':>10}")
    for tau, Z in zip(args.taus, frames):
        d = sd.dipole_field(mdl, Z)
        print(f"{tau:>8g} {sd.energy(mdl, Z) / mdl.L:>12.6f} "
              f"{d[:, 2].mean():>9.4f} {np.linalg.norm(d, axis=1).mean():>9.4f} "
              f"{sd.sun_topological_charge(Z, tris):>10.4f}")

    plot(frames, args.taus, args.L, a1, a2, tris, args.out)


if __name__ == "__main__":
    main()
