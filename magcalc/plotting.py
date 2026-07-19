import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
from typing import Optional, List, Tuple, Union

logger = logging.getLogger(__name__)


def broaden_spectrum(
    centers: np.ndarray,
    weights: np.ndarray,
    eval_grid: np.ndarray,
    width: Union[float, np.ndarray] = 0.2,
    kind: str = "lorentzian",
) -> np.ndarray:
    """
    Broaden a set of delta-function modes into a continuous spectrum.

    Given mode energies ``centers`` with spectral weights ``weights`` (e.g. the
    per-mode S(Q,w) intensities at one Q-point), return the broadened intensity
    sampled at every energy in ``eval_grid``. This is the single source of truth
    for energy broadening, shared by the S(Q,w) plot and the intensity-fitting
    residuals (so the fit and the plot use identical line-shapes).

    Args:
        centers: (M,) mode energies for one Q-point. NaN/complex entries are dropped.
        weights: (M,) spectral weights (intensities) for those modes.
        eval_grid: (G,) energies at which to evaluate the broadened spectrum.
        width: full-width parameter (FWHM) of the line-shape, in meV. Either a
            scalar (same width for every mode) or an (M,) array of per-mode
            widths (energy-dependent instrument resolution).
        kind: "lorentzian" (default, matches the S(Q,w) map) or "gaussian".

    Returns:
        (G,) array of broadened intensity at each ``eval_grid`` energy.
    """
    centers = np.asarray(centers, dtype=float).ravel() if not np.iscomplexobj(centers) \
        else np.real(np.asarray(centers).ravel())
    weights = np.asarray(weights, dtype=float).ravel() if not np.iscomplexobj(weights) \
        else np.real(np.asarray(weights).ravel())
    eval_grid = np.asarray(eval_grid, dtype=float).ravel()

    out = np.zeros(eval_grid.shape[0], dtype=float)
    if centers.size == 0:
        return out

    width_arr = np.asarray(width, dtype=float)
    if width_arr.ndim == 0:
        width_arr = np.full(centers.shape, float(width_arr))
    else:
        width_arr = np.broadcast_to(width_arr.ravel(), centers.shape).copy()

    valid = ~np.isnan(centers) & ~np.isnan(weights) & ~np.isnan(width_arr)
    centers = centers[valid]
    weights = weights[valid]
    width_arr = width_arr[valid]
    if centers.size == 0:
        return out

    width_arr = np.maximum(width_arr, 1e-9)
    delta = eval_grid[:, None] - centers[None, :]  # (G, M)
    if kind == "gaussian":
        sigma = width_arr[None, :] / 2.3548200450309493  # FWHM -> sigma
        shape = np.exp(-0.5 * (delta / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    else:  # lorentzian
        hwhm = width_arr[None, :] / 2.0
        shape = (1.0 / np.pi) * hwhm / (delta ** 2 + hwhm ** 2)

    return (shape * weights[None, :]).sum(axis=1)


# E(meV) = HBAR2_2M * k^2(1/A^2) for a neutron.
HBAR2_2M_MEV_A2 = 2.0721


def kinematic_q_bounds(
    e_transfer: np.ndarray,
    ei_mev: float,
    two_theta_deg: Tuple[float, float] = (0.0, 180.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Accessible |Q| range for a direct-geometry spectrometer.

    For incident energy Ei and energy transfer w, ki = sqrt(Ei/C),
    kf = sqrt((Ei-w)/C) with C = 2.0721 meV A^2, and the detector coverage
    two_theta_deg limits |Q| to
        Q(w, tth) = sqrt(ki^2 + kf^2 - 2 ki kf cos(tth)).
    Returns (q_min, q_max) arrays; both are NaN where w > Ei (kinematically
    forbidden).
    """
    w = np.asarray(e_transfer, dtype=float)
    ei = float(ei_mev)
    ki = np.sqrt(ei / HBAR2_2M_MEV_A2)
    with np.errstate(invalid="ignore"):
        kf = np.sqrt(np.where(w <= ei, (ei - w) / HBAR2_2M_MEV_A2, np.nan))
    tth_lo, tth_hi = np.deg2rad(two_theta_deg[0]), np.deg2rad(two_theta_deg[1])
    q_lo = np.sqrt(np.maximum(ki**2 + kf**2 - 2 * ki * kf * np.cos(tth_lo), 0.0))
    q_hi = np.sqrt(np.maximum(ki**2 + kf**2 - 2 * ki * kf * np.cos(tth_hi), 0.0))
    return np.minimum(q_lo, q_hi), np.maximum(q_lo, q_hi)


def resolve_de_fwhm(
    energies: np.ndarray,
    resolution: Optional[dict],
    default_width: float,
) -> Union[float, np.ndarray]:
    """Per-mode energy FWHM from a plotting.resolution block.

    resolution['de_fwhm'] may be a scalar (constant FWHM) or a polynomial
    coefficient list evaluated with numpy.polyval (HIGHEST power first — the
    SpinW sw_instrument 'dE' convention): FWHM(E) = polyval(coeffs, E).
    Falls back to default_width when absent. Non-positive evaluations are
    clipped to 1e-4 meV.
    """
    if not resolution or resolution.get("de_fwhm") is None:
        return default_width
    de = resolution["de_fwhm"]
    if np.isscalar(de):
        return float(de)
    coeffs = np.asarray(de, dtype=float)
    widths = np.polyval(coeffs, np.asarray(energies, dtype=float))
    return np.maximum(widths, 1e-4)


def plot_dispersion(
    q_vectors: np.ndarray,
    energies: Union[List[np.ndarray], np.ndarray],
    save_filename: str,
    title: str = "Spin Wave Dispersion",
    ylim: Optional[List[float]] = None,
    show_plot: bool = False,
    auto_scale: bool = True
):
    """
    Plots the spin-wave dispersion relation.

    An explicit ``ylim`` is always honoured. When ``ylim`` is None the y-axis
    limits are derived from the computed mode energies. (``auto_scale`` is kept
    for API compatibility; it no longer overrides an explicit ``ylim`` — that
    made `plotting.energy_limits_disp` silently ineffective.)
    """
    try:
        # Calculate path length
        if len(q_vectors) > 0:
            diffs = np.diff(q_vectors, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            path_len = np.concatenate(([0], np.cumsum(dists)))
            x_vals = path_len
        else:
            x_vals = []

        # Convert energies list to array if needed (handling irregular shapes if necessary, 
        # but usually N_modes is constant)
        # Using simple iteration for potential ragged arrays
        
        plt.figure(figsize=(8, 6))
        
        # Plot each q-point's modes
        # Optimizing: if energies is regular (N, M), plot as lines.
        # If irregular, scatter.
        is_regular = False
        if isinstance(energies, np.ndarray) and energies.ndim == 2:
            is_regular = True
        elif isinstance(energies, list) and len(energies) > 0 and isinstance(energies[0], (list, np.ndarray)):
             lens = [len(e) for e in energies]
             if len(set(lens)) == 1:
                 is_regular = True
                 energies = np.array(energies)

        if is_regular:
            # energies shape (N_q, N_modes)
            # Plot lines
            for mode_idx in range(energies.shape[1]):
                plt.plot(x_vals, energies[:, mode_idx], 'b-', alpha=0.8)
        else:
            # Scatter for ragged
            for i, x in enumerate(x_vals):
                ens = energies[i]
                plt.scatter([x]*len(ens), ens, c='b', s=10, alpha=0.6)

        plt.title(title)
        plt.xlabel(r"Q Path Length ($\AA^{-1}$)")
        plt.ylabel("Energy (meV)")

        # Auto-scale the y-axis from the mode energies only when no explicit
        # limits were supplied.
        if ylim is None:
            all_ens = []
            if isinstance(energies, np.ndarray):
                all_ens = energies.flatten().tolist()
            else:
                for e in energies:
                    all_ens.extend(np.asarray(e).flatten().tolist())

            valid_ens = []
            for e in all_ens:
                if isinstance(e, complex) or type(e).__name__.startswith('complex'):
                    if abs(e.imag) > 1e-3:
                        continue
                    e = e.real
                e = float(e)
                if np.isnan(e) or np.isinf(e):
                    continue
                valid_ens.append(e)

            if valid_ens:
                data_min, data_max = min(valid_ens), max(valid_ens)
                span = data_max - data_min
                pad = 0.05 * span if span > 0 else max(abs(data_max), 1.0) * 0.05
                y_top = data_max + pad
                y_bottom = min(0.0, data_min - pad)
                ylim = [y_bottom, y_top]

        if ylim:
            plt.ylim(ylim)
        plt.grid(True, alpha=0.3)
        plt.xlim(min(x_vals), max(x_vals))

        # Ensure directory exists
        if save_filename:
             os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
             plt.tight_layout()
             plt.savefig(save_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
             logger.info(f"Dispersion plot saved to {save_filename}")
        
        if show_plot:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot dispersion: {e}")
        raise e

def plot_sqw_map(
    q_vectors: np.ndarray,
    energies: Union[List[np.ndarray], np.ndarray],
    intensities: Union[List[np.ndarray], np.ndarray],
    save_filename: str,
    title: str = "S(Q,w)",
    ylim: Optional[List[float]] = None,
    broadening_width: float = 0.2,
    cmap: str = 'PuBu_r',
    show_plot: bool = False,
    resolution: Optional[dict] = None,
    x_is_qmag: bool = False,
    energy_grid_step: float = 0.05
):
    """
    Plots the S(Q,w) intensity map with Gaussian/Lorentzian broadening.

    resolution (optional dict) models the instrument (SpinW sw_instrument
    analogue):
      de_fwhm: scalar FWHM (meV) or polyval coefficient list (highest power
          first) for an energy-dependent FWHM(E). Overrides broadening_width.
      shape: 'gaussian' | 'lorentzian'. Defaults to gaussian when de_fwhm is
          given (instrument resolution), else the legacy lorentzian.
      dq_fwhm: Gaussian FWHM (1/A) smoothing along the x axis (q path length
          or |Q| — both are in 1/A).
      ei: incident energy (meV, direct geometry). Masks energy transfer > Ei.
      two_theta: [min, max] detector coverage (deg). With ei, and when the x
          axis is |Q| (x_is_qmag=True, i.e. powder maps), masks |Q| outside
          the kinematically accessible range at each energy.
    Masked cells are NaN (blank in the map).
    """
    try:
        # 1. Path Length
        if len(q_vectors) > 0:
            diffs = np.diff(q_vectors, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            path_len = np.concatenate(([0], np.cumsum(dists)))
            x_vals = path_len
        else:
            x_vals = np.arange(len(q_vectors))

        # 2. Grid Setup
        if ylim is None:
            # Auto-detect max energy?
             # Flatten energies to find max
             all_ens = []
             if isinstance(energies, np.ndarray): all_ens = energies.flatten()
             else: 
                 for e in energies: all_ens.extend(e)
                 
             valid_ens = []
             for e in all_ens:
                 if np.isnan(e): continue
                 if isinstance(e, complex) or type(e).__name__.startswith('complex'):
                     if abs(e.imag) > 1e-3: continue
                     e = e.real
                 valid_ens.append(float(e))
                 
             if not valid_ens:
                 y_max = 20
             else:
                 y_max = max(valid_ens) * 1.1
             y_min = 0
             ylim = [y_min, y_max]
        
        y_min, y_max = ylim
        dy = float(energy_grid_step) # Energy grid step
        y_grid = np.arange(y_min, y_max + dy, dy)

        # Line shape: instrument resolution blocks default to gaussian,
        # the legacy constant-width path stays lorentzian.
        shape_kind = (resolution or {}).get('shape')
        if shape_kind is None:
            shape_kind = 'gaussian' if (resolution or {}).get('de_fwhm') is not None \
                else 'lorentzian'

        # 3. Broadening (shared line-shape with the intensity-fitting residuals)
        intensity_matrix = np.zeros((len(y_grid), len(x_vals)))

        for i_q in range(len(x_vals)):
            ens = energies[i_q]
            ints = intensities[i_q]

            # Handle None or NaN
            if ens is None or ints is None: continue

            widths = resolve_de_fwhm(np.asarray(ens, dtype=float).ravel(),
                                     resolution, broadening_width)
            intensity_matrix[:, i_q] = broaden_spectrum(
                ens, ints, y_grid, width=widths, kind=shape_kind
            )

        # 3b. Instrument effects: dQ smoothing and Ei kinematic masking.
        if resolution:
            dq_fwhm = resolution.get('dq_fwhm')
            if dq_fwhm and len(x_vals) > 1:
                from scipy.ndimage import gaussian_filter1d
                dx = float(np.median(np.diff(x_vals)))
                if dx > 0:
                    sigma_px = (float(dq_fwhm) / 2.3548200450309493) / dx
                    intensity_matrix = gaussian_filter1d(
                        intensity_matrix, sigma_px, axis=1, mode='nearest')

            ei = resolution.get('ei')
            if ei is not None:
                ei = float(ei)
                intensity_matrix[y_grid > ei, :] = np.nan
                two_theta = resolution.get('two_theta')
                if two_theta is not None and x_is_qmag:
                    q_lo, q_hi = kinematic_q_bounds(
                        y_grid, ei, (float(two_theta[0]), float(two_theta[1])))
                    x_arr = np.asarray(x_vals, dtype=float)[None, :]
                    forbidden = (x_arr < q_lo[:, None]) | (x_arr > q_hi[:, None]) \
                        | ~np.isfinite(q_lo)[:, None]
                    intensity_matrix[forbidden] = np.nan

        # 4. Plotting
        plt.figure(figsize=(10, 6))

        # Robust Vmin/Vmax
        finite = np.isfinite(intensity_matrix)
        pos_vals = intensity_matrix[finite & (intensity_matrix > 1e-6)]
        if len(pos_vals) > 0:
            vmin = np.min(pos_vals)
            vmax = np.max(pos_vals)
        else:
            vmin, vmax = 1e-3, 1.0

        pcm = plt.pcolormesh(x_vals, y_grid, intensity_matrix, 
                             norm=LogNorm(vmin=vmin, vmax=vmax),
                             cmap=cmap,
                             shading='nearest') 
                             
        plt.colorbar(pcm, label="Intensity (arb. units)")
        plt.title(title)
        plt.xlabel(r"$|Q|$ ($\AA^{-1}$)" if x_is_qmag else r"Q Path Length ($\AA^{-1}$)")
        plt.ylabel("Energy (meV)")
        plt.ylim(ylim)
        plt.xlim(min(x_vals), max(x_vals))
        
        plt.tight_layout()
        
        if save_filename:
             os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
             # tight_layout is already called above, but good to ensure
             plt.savefig(save_filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
             logger.info(f"S(Q,w) plot saved to {save_filename}")

        if show_plot:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot S(Q,w): {e}")
        raise e


def plot_energy_cuts(
    coords1: np.ndarray,
    coords2: np.ndarray,
    panels: List[np.ndarray],
    labels: List[str],
    save_filename: str,
    axis_labels: Tuple[str, str] = ("axis 1 (r.l.u.)", "axis 2 (r.l.u.)"),
    title: str = "Constant-energy cuts",
    cmap: str = "viridis",
    show_plot: bool = False,
):
    """Plot one panel per constant-energy cut on a 2-D q grid.

    coords1/coords2 are the grid axis coordinates (n1,) and (n2,); each panel
    is an (n1, n2) intensity array (axis 0 <-> coords1). Color scale is capped
    at the 97th percentile of the positive values, like the SW10 reference.
    """
    try:
        n_panels = max(len(panels), 1)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(5.5 * n_panels, 4.6),
            constrained_layout=True, squeeze=False)
        for ax, Z, label in zip(axes[0], panels, labels):
            Zp = np.asarray(Z, dtype=float)
            finite_pos = Zp[np.isfinite(Zp) & (Zp > 0)]
            vmax = np.percentile(finite_pos, 97) if finite_pos.size else 1.0
            pm = ax.pcolormesh(coords1, coords2, Zp.T, shading="auto",
                               cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_title(label)
            fig.colorbar(pm, ax=ax, shrink=0.85)
        fig.suptitle(title)

        if save_filename:
            os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
            fig.savefig(save_filename, dpi=150, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"Energy-cut plot saved to {save_filename}")
        if show_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot energy cuts: {e}")
        raise e


def plot_fit_comparison(
    fit_type: str,
    prediction: dict,
    save_filename: str,
    title: str = "Fit comparison",
    show_plot: bool = False,
):
    """
    Plot a data-vs-best-fit-model comparison produced by ``FitProblem.predict``.

    dispersion: measured peak energies (black, with error bars) overlaid with the
        matched model energies (red), plus a model-vs-data parity panel.
    sqw/powder: a model-vs-data intensity parity scatter and a residual panel,
        which work for both scattered and gridded data.
    """
    try:
        if fit_type == "dispersion":
            x = prediction["x"]
            E_data = prediction["E_data"]
            sigma = prediction["sigma"]
            E_model = prediction["E_model"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.errorbar(x, E_data, yerr=sigma, fmt="ko", ms=4, label="data", capsize=2)
            ax1.plot(x, E_model, "rx", ms=7, label="best-fit model")
            ax1.set_xlabel("data point index")
            ax1.set_ylabel("Energy (meV)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            lo = float(min(E_data.min(), E_model.min()))
            hi = float(max(E_data.max(), E_model.max()))
            ax2.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
            ax2.errorbar(E_data, E_model, xerr=sigma, fmt="bo", ms=4, capsize=2)
            ax2.set_xlabel("measured E (meV)")
            ax2.set_ylabel("model E (meV)")
            ax2.set_title("parity")
            ax2.grid(True, alpha=0.3)
        else:
            I_data = prediction["I_data"]
            I_model = prediction["I_model"]
            resid = I_model - I_data

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            lo = float(min(I_data.min(), I_model.min()))
            hi = float(max(I_data.max(), I_model.max()))
            ax1.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
            ax1.scatter(I_data, I_model, s=10, alpha=0.6)
            ax1.set_xlabel("measured intensity")
            ax1.set_ylabel("model intensity")
            ax1.set_title("parity")
            ax1.grid(True, alpha=0.3)

            ax2.scatter(prediction["energy"], resid, s=10, alpha=0.6)
            ax2.axhline(0.0, color="k", ls="--", alpha=0.6)
            ax2.set_xlabel("Energy (meV)")
            ax2.set_ylabel("model - data")
            ax2.set_title("residuals")
            ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        fig.tight_layout()

        if save_filename:
            os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
            fig.savefig(save_filename, dpi=150, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"Fit comparison plot saved to {save_filename}")
        if show_plot:
            plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"Failed to plot fit comparison: {e}")
        raise e


def plot_scga(intensities, save_filename=None, temperature=None, labels=None,
              show_plot=False):
    """SCGA diffuse S(q) along the q-path (one line, q-path index on x)."""
    try:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(intensities))
        ax.plot(x, intensities, "-", lw=1.8)
        ax.set_xlabel("q-path index")
        ax.set_ylabel("S(q) (arb. units)")
        ttl = "SCGA diffuse scattering"
        if temperature is not None:
            ttl += f"  (kT = {temperature:g} meV)"
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_filename:
            os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
            fig.savefig(save_filename, dpi=150, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"SCGA plot saved to {save_filename}")
        if show_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot SCGA S(q): {e}")
        raise


def plot_thermal_mc(temperatures, energy, heat_capacity, magnetization,
                    susceptibility, save_filename=None, show_plot=False):
    """Thermal-MC thermodynamics: 2x2 panels of <E>/N, C/N, |M|/NS, chi vs kT."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharex=True)
        panels = [(energy, r"$\langle E\rangle/N$ (meV)"),
                  (heat_capacity, r"$C/N$"),
                  (magnetization, r"$|M|/(NS)$"),
                  (susceptibility, r"$\chi$ per spin")]
        for ax, (y, lbl) in zip(axes.ravel(), panels):
            ax.plot(temperatures, y, "o-", ms=4)
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("kT (meV)")
        fig.suptitle("Thermal Monte-Carlo (parallel tempering)")
        fig.tight_layout()
        if save_filename:
            os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
            fig.savefig(save_filename, dpi=150, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"Thermal-MC plot saved to {save_filename}")
        if show_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot thermal MC: {e}")
        raise


def plot_sqw_grid(energies, intensities, save_filename=None, title="S(q,w)",
                  e_max=None, log_scale=True, show_plot=False):
    """Generic S(q,w) map on (q-path index, energy) -- used by the KPM and
    SampledCorrelations tasks. `intensities` has shape (n_energies, n_q)."""
    try:
        energies = np.asarray(energies, float)
        I = np.asarray(intensities, float)
        if e_max is not None:
            keep = energies <= float(e_max)
            energies, I = energies[keep], I[keep]
        fig, ax = plt.subplots(figsize=(7, 4.8))
        pos = I[I > 0]
        norm = None
        if log_scale and pos.size:
            vmax = float(pos.max())
            norm = LogNorm(vmin=max(vmax * 1e-4, float(pos.min())), vmax=vmax)
        mesh = ax.pcolormesh(np.arange(I.shape[1] + 1), _bin_edges(energies), I,
                             cmap="viridis", norm=norm, shading="flat")
        fig.colorbar(mesh, ax=ax, label="Intensity (arb. units)")
        ax.set_xlabel("q-path index")
        ax.set_ylabel("Energy (meV)")
        ax.set_title(title)
        fig.tight_layout()
        if save_filename:
            os.makedirs(os.path.dirname(os.path.abspath(save_filename)), exist_ok=True)
            fig.savefig(save_filename, dpi=150, bbox_inches="tight", pad_inches=0.1)
            logger.info(f"S(q,w) map saved to {save_filename}")
        if show_plot:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to plot S(q,w) map: {e}")
        raise


def _bin_edges(centers):
    centers = np.asarray(centers, float)
    if len(centers) < 2:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = 0.5 * (centers[1:] + centers[:-1])
    return np.concatenate([[centers[0] - (mid[0] - centers[0])], mid,
                           [centers[-1] + (centers[-1] - mid[-1])]])
