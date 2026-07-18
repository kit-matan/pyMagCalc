"""Powder comparison for Cu5SbO6: experiment vs the paper's analytic model vs pyMagCalc.

Produces `powder_compare.png` in this folder -- three panels of the powder-averaged
triplon S(|Q|, w), all on the SAME experimental (|Q|, E) grid, all rendered with the
SAME smoothing and colour scale, so only the physics differs:

  left   -- EXPERIMENT: the low-T triplon map with the paramagnetic 200 K background
            subtracted (analytic/map_triplons_sub.csv);
  centre -- the manuscript's first-order dimer expansion (analytic/powderINS_sim.py),
            read back from analytic/simulated_map.csv;
  right  -- pyMagCalc's `mode: entangled` engine (config.yaml), the full harmonic
            (Sachdev-Bhatt) sqrt-resummation of that same expansion.

Both simulations use the Lorentzian width eps = J1/40 and the paper's Cu2+
form-factor coefficients; pyMagCalc's spherical average is converged with 1500
Fibonacci directions per |Q| shell. All three maps are regridded onto one common
mesh and passed through the same Gaussian filter. Each panel is normalized to the
99th percentile INSIDE the triplon band (E in [9, 21] meV) so the experiment's
residual elastic line (E->0) and kinematic-edge artifacts do not set the scale.

Run (needs analytic/simulated_map.csv from analytic/powderINS_sim.py, and
analytic/map_triplons_sub.csv):
    python examples/entangled/Cu5SbO6/powder_compare.py

Reference: Piyakulworawat et al., Phys. Rev. Research 8, 013247 (2026).
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.entangled import EntangledCalculator
from magcalc.numerical import powder_sample_modes
from magcalc.form_factors import get_form_factor

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYTIC = os.path.join(HERE, "analytic")

# --- Broadening / form factor: identical to analytic/powderINS_sim.py ----------
J1 = 16.5
EPS = J1 / 40.0                                   # Lorentzian HWHM (meV)
FF = dict(A=0.0232, B=0.1502, C=0.8305, D=-0.0039, a=34.488, b=13.113, c=5.392)


def form_factor(q):
    """Cu2+ magnetic form factor F(|Q|), the paper's <j0> parameterization."""
    s = q / (4 * np.pi)
    return (FF["A"] * np.exp(-FF["a"] * s ** 2)
            + FF["B"] * np.exp(-FF["b"] * s ** 2)
            + FF["C"] * np.exp(-FF["c"] * s ** 2)
            + FF["D"])


# --- Manuscript model: the (|Q|, E) grid and its simulated intensities ---------
sim = np.loadtxt(os.path.join(ANALYTIC, "simulated_map.csv"),
                 delimiter=",", skiprows=1)
Q, E, I_manuscript = sim[:, 0], sim[:, 1], sim[:, 2]

# --- Experiment: the low-T triplon map with the paramagnetic (200 K) background
# subtracted (analytic/map_triplons_sub.csv), on the SAME (|Q|, E) grid. This is
# the measured triplon band the simulations are compared against.
exp = np.genfromtxt(os.path.join(ANALYTIC, "map_triplons_sub.csv"),
                    skip_header=1)
Qx, Ex, I_exp = exp[:, 0], exp[:, 1], exp[:, 2]

# --- pyMagCalc entangled engine on the SAME (|Q|, E) grid ----------------------
cfg_path = os.path.join(HERE, "config.yaml")
with open(cfg_path) as f:
    config = yaml.safe_load(f)
model = GenericSpinModel(config, base_path=HERE)
config = model.config
params = [config["parameters"][k] for k in config["parameter_order"]]
calc = EntangledCalculator(model, config, params)

# Converge the spherical average: 1500 Fibonacci directions per |Q| shell keeps
# each direction's own triplon energy (the correct powder lineshape) while
# removing the directional-sampling texture.
uQ = np.unique(Q)
E_modes, I_modes = powder_sample_modes(calc, uQ, num_samples=1500,
                                       cross_section="perp")
q_to_shell = {q: i for i, q in enumerate(uQ)}

# calculate_sqw ALREADY multiplies the intensity by the Cu2+ magnetic form factor
# internally (ion: Cu2+ in the config -> get_form_factor in lswt.structure_factor),
# together with the staggered dimer (1 - cos q.d) structure factor. So the |Q|
# falloff is already baked into I_modes -- multiplying by form_factor(q)**2 again
# would apply it TWICE and over-steepen the pyMagCalc falloff. To compare against
# the manuscript on an equal footing, divide out pyMagCalc's built-in form factor
# (Int-Tables Cu2+) and reapply the manuscript's parameterization, so BOTH panels
# carry F(Q)**2 exactly once, with the SAME coefficients -- only the dispersion
# (sqrt-resummed vs first-order) then differs.
I_pymagcalc = np.zeros_like(Q)
for j, (q, e) in enumerate(zip(Q, E)):
    Em = E_modes[q_to_shell[q]]
    Im = I_modes[q_to_shell[q]]
    good = np.isfinite(Em) & np.isfinite(Im)
    Em, Im = Em[good], Im[good]
    lorentz = (EPS / np.pi) / ((e - Em) ** 2 + EPS ** 2)
    ff_ratio = (form_factor(q) / get_form_factor("Cu2+", q)) ** 2
    I_pymagcalc[j] = ff_ratio * np.sum(Im * lorentz)

# --- Identical smoothing: regrid both onto one mesh, same Gaussian filter -------
q_grid = np.linspace(0.3, 5.6, 220)
e_grid = np.linspace(0.0, 21.0, 220)
QG, EG = np.meshgrid(q_grid, e_grid)


def grid_and_smooth(q, e, intensity, sigma=1.6):
    G = griddata((q, e), intensity, (QG, EG), method="linear")
    outside = np.isnan(G)
    G = gaussian_filter(np.nan_to_num(G, nan=0.0), sigma=sigma)
    G[outside] = np.nan                           # keep the kinematic mask blank
    return G


def normalize(G, e_ref=(9.0, 21.0)):
    # Scale to the 99th percentile INSIDE the triplon band so the residual
    # elastic line (E->0) and kinematic-edge artifacts don't set the colour
    # scale. Same reference window for every panel.
    band = (EG >= e_ref[0]) & (EG <= e_ref[1])
    return np.clip(G / np.nanpercentile(G[band], 99), 0, 1)


panels = [("Experiment (200 K subtracted)", grid_and_smooth(Qx, Ex, I_exp)),
          ("Manuscript powderINS_sim.py", grid_and_smooth(Q, E, I_manuscript)),
          ("pyMagCalc (entangled engine)", grid_and_smooth(Q, E, I_pymagcalc))]

fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for ax, (title, G) in zip(axs, panels):
    im = ax.pcolormesh(q_grid, e_grid, normalize(G),
                       cmap="viridis", shading="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel(r"|Q| ($\AA^{-1}$)")
    ax.set_xlim(0.3, 5.6)
    ax.set_ylim(0, 21)
    fig.colorbar(im, ax=ax, label="I (norm.)")
axs[0].set_ylabel("Energy transfer (meV)")
plt.tight_layout()

out = os.path.join(HERE, "powder_compare.png")
plt.savefig(out, dpi=120)
print(f"Saved comparison figure to {out}")
