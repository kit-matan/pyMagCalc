"""SW10 - constant-energy cut on the square-lattice Neel AFM.

Port of SpinW Tutorial 10 / Sunny "SW10" to pyMagCalc. Loads config.yaml
(2x2 Neel cell, J = 1 meV, S = 1), computes S(q,w) on an (H,K) grid over
chemical RLU [0,2]^2 at L = 0, and reproduces the tutorial's two panels:

  1. constant-energy cut at E = 3.75 meV, Gaussian FWHM 0.2 meV;
  2. intensities integrated over 3.5 <= E < 4.01 meV.

Run from the pyMagCalc repo root:
    python examples/spinw_tutorials/SW10_energy_cut/sw10_energy_cut.py
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.runner import compute_b_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NGRID = 261          # grid points per axis (chemical RLU 0..2)
E_CUT = 3.75         # meV
FWHM = 0.25          # meV
E_BAND = (3.5, 4.01) # integration window (meV)


def main():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    model = GenericSpinModel(cfg)
    thetas, phis = model.generate_magnetic_structure()
    model.set_magnetic_structure(thetas, phis)
    B = compute_b_matrix(model)

    calc = mc.MagCalc(spin_model_module=model, spin_magnitude=1.0,
                      hamiltonian_params=[1.0], cache_file_base=os.path.join(HERE, ".sw10"),
                      cache_mode="none")

    # (H,K) grid in CHEMICAL RLU; the 2x2 magnetic cell doubles the RLU.
    hs = np.linspace(0.0, 2.0, NGRID)
    H, K = np.meshgrid(hs, hs, indexing="ij")
    q_rlu = np.column_stack([2.0 * H.ravel(), 2.0 * K.ravel(), np.zeros(H.size)])
    q_cart = q_rlu @ B

    res = calc.calculate_sqw(q_cart)
    E = np.asarray(res.energies)        # (Nq, modes)
    I = np.asarray(res.intensities)     # (Nq, modes)

    sigma = FWHM / 2.3548200450309493
    cut = np.sum(I * np.exp(-((E - E_CUT) ** 2) / (2 * sigma ** 2)), axis=1)
    band = np.sum(I * ((E >= E_BAND[0]) & (E < E_BAND[1])), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, Z, title in [
        (axes[0], cut.reshape(NGRID, NGRID),
         f"Constant-E cut at {E_CUT} meV (FWHM {FWHM} meV)"),
        (axes[1], band.reshape(NGRID, NGRID),
         f"Integrated {E_BAND[0]}-{E_BAND[1]} meV"),
    ]:
        vmax = np.percentile(Z[Z > 0], 97) if np.any(Z > 0) else 1.0
        pm = ax.pcolormesh(hs, hs, Z.T, shading="auto", cmap="viridis",
                           vmin=0.0, vmax=vmax)
        ax.set_xlabel("H (chemical r.l.u.)")
        ax.set_ylabel("K (chemical r.l.u.)")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(pm, ax=ax, shrink=0.85)
    out = os.path.join(HERE, "SW10_cuts.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
