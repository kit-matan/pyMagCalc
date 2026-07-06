import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


def broaden_spectrum(
    centers: np.ndarray,
    weights: np.ndarray,
    eval_grid: np.ndarray,
    width: float = 0.2,
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
        width: full-width parameter (FWHM) of the line-shape, in meV.
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

    valid = ~np.isnan(centers) & ~np.isnan(weights)
    centers = centers[valid]
    weights = weights[valid]
    if centers.size == 0:
        return out

    width = max(float(width), 1e-9)
    delta = eval_grid[:, None] - centers[None, :]  # (G, M)
    if kind == "gaussian":
        sigma = width / 2.3548200450309493  # FWHM -> sigma
        shape = np.exp(-0.5 * (delta / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    else:  # lorentzian
        hwhm = width / 2.0
        shape = (1.0 / np.pi) * hwhm / (delta ** 2 + hwhm ** 2)

    return (shape * weights[None, :]).sum(axis=1)


def plot_dispersion(
    q_vectors: np.ndarray,
    energies: Union[List[np.ndarray], np.ndarray],
    save_filename: str,
    title: str = "Spin Wave Dispersion",
    ylim: Optional[List[float]] = None,
    show_plot: bool = False
):
    """
    Plots the spin-wave dispersion relation.
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
    show_plot: bool = False
):
    """
    Plots the S(Q,w) intensity map with Gaussian/Lorentzian broadening.
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
        dy = 0.05 # Energy resolution
        y_grid = np.arange(y_min, y_max + dy, dy)
        
        # 3. Broadening (shared line-shape with the intensity-fitting residuals)
        intensity_matrix = np.zeros((len(y_grid), len(x_vals)))

        for i_q in range(len(x_vals)):
            ens = energies[i_q]
            ints = intensities[i_q]

            # Handle None or NaN
            if ens is None or ints is None: continue

            intensity_matrix[:, i_q] = broaden_spectrum(
                ens, ints, y_grid, width=broadening_width, kind="lorentzian"
            )

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        
        # Robust Vmin/Vmax
        pos_vals = intensity_matrix[intensity_matrix > 1e-6]
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
        plt.xlabel(r"Q Path Length ($\AA^{-1}$)")
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
