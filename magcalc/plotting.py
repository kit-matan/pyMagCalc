import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import os
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

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
             plt.savefig(save_filename, dpi=150)
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
             all_ens = [e for e in all_ens if not np.isnan(e)]
             if not all_ens:
                 y_max = 20
             else:
                 y_max = max(all_ens) * 1.1
             y_min = 0
             ylim = [y_min, y_max]
        
        y_min, y_max = ylim
        dy = 0.05 # Energy resolution
        y_grid = np.arange(y_min, y_max + dy, dy)
        
        # 3. Broadening
        intensity_matrix = np.zeros((len(y_grid), len(x_vals)))
        
        for i_q in range(len(x_vals)):
            ens = energies[i_q]
            ints = intensities[i_q]
            
            # Handle None or NaN
            if ens is None or ints is None: continue
            if isinstance(ens, (list, tuple)): ens = np.array(ens)
            if isinstance(ints, (list, tuple)): ints = np.array(ints)
            
            valid = ~np.isnan(ens) & ~np.isnan(ints)
            ens = ens[valid]
            ints = ints[valid]
            
            if len(ens) == 0: continue
            
            # Lorentzian Broadening
            for band_idx, en_val in enumerate(ens):
                w_val = ints[band_idx]
                denom = (y_grid - en_val)**2 + (broadening_width/2)**2
                lor = (1.0 / np.pi) * (broadening_width / 2) / denom
                intensity_matrix[:, i_q] += w_val * lor

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
             plt.savefig(save_filename, dpi=150)
             logger.info(f"S(Q,w) plot saved to {save_filename}")

        if show_plot:
            plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot S(Q,w): {e}")
        raise e
