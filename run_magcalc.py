import sys
import os
import yaml
import logging
import magcalc as mc
import numpy as np
from generic_model import GenericSpinModel
import matplotlib.pyplot as plt
import matplotlib # Ensure backend setting if needed, though usually fine on mac

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def generate_q_path_from_config(config):
    q_conf = config.get('q_path', {})
    points = {k: np.array(v) for k, v in q_conf.items() if k not in ['path', 'points_per_segment', 'unit']}
    path_labels = q_conf.get('path', [])
    n_points = q_conf.get('points_per_segment', 50)
    
    q_vectors = []
    # If path is list of strings
    if path_labels and len(path_labels) > 1:
        for i in range(len(path_labels)-1):
            start_label = path_labels[i]
            end_label = path_labels[i+1]
            
            start_pt = points.get(start_label)
            end_pt = points.get(end_label)
            
            if start_pt is None or end_pt is None:
                 logger.error(f"Undefined point in path: {start_label} -> {end_label}")
                 continue
                 
            # Linear interpolation
            # Use endpoint=False for intermediate segments to avoid doubles?
            # Or handle explicitly.
            # Using endpoint=True and slicing later.
            segment = np.linspace(start_pt, end_pt, n_points)
            if i > 0:
                segment = segment[1:] # Avoid duplicate points at junctions
            q_vectors.extend(segment)
            
    return np.array(q_vectors)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_magcalc.py <config.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    # Load Config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    
    # Initialize Generic Model
    # Determine base path relative to config file for CIF loading
    config_dir = os.path.dirname(os.path.abspath(config_file))
    spin_model = GenericSpinModel(config, base_path=config_dir)
    
    # Initialize MagCalc
    calc_config = config.get('calculation', {})
    cache_mode = calc_config.get('cache_mode', 'r')
    cache_base = calc_config.get('cache_file_base', 'magcalc_cache')
    
    # We might need to handle parameters if MagCalc needs them separately? 
    # MagCalc takes (spin_model_module, params_val, ...)
    # GenericSpinModel needs params passed to Hamiltonian.
    # We need to construct params_val list based on 'parameters' order in config.
    
    params_val = []
    param_names = config.get('parameters', [])
    model_params = config.get('model_params', {})
    
    if param_names:
        logger.info(f"Using parameter order: {param_names}")
        for name in param_names:
            val = model_params.get(name)
            if val is None:
                logger.warning(f"Parameter '{name}' not found in 'model_params'. Defaulting to 0.0")
                val = 0.0
            params_val.append(float(val))
    
    # Extract Spin Magnitude
    S_val = model_params.get('S')
    if S_val is None:
         logger.warning("Parameter 'S' (spin magnitude) not found in 'model_params'. Defaulting to 1.0.")
         S_val = 1.0
    S_val = float(S_val)

    logger.info(f"Model Parameters: {params_val}, S={S_val}")
    
    # Initialize Calculator
    calculator = mc.MagCalc(
        spin_model_module=spin_model, 
        spin_magnitude=S_val,
        hamiltonian_params=params_val,
        cache_file_base=cache_base,
        cache_mode=cache_mode
    )
    
    # Run Tasks
    tasks = config.get('tasks', {})
    q_vectors = None
    
    # 1. Dispersion
    if tasks.get('run_dispersion', False):
        logger.info("--- Starting Dispersion Workflow ---")
        disp_file = config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
        if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
        
        # Check if we need to calculate
        need_recalc = tasks.get('calculate_dispersion_new', True) or not os.path.exists(disp_file)
        
        if need_recalc:
             # Generate Q vectors if not yet generated
             if q_vectors is None:
                 q_vectors = generate_q_path_from_config(config)
                 if len(q_vectors) == 0:
                     logger.warning("No Q-vectors generated from config 'q_path'. Skipping calculation.")
                 else:
                     logger.info(f"Generated {len(q_vectors)} Q-vectors (RLU) for calculation.")
                     
                     # Convert RLU to Cartesian
                     try:
                         uc = spin_model.unit_cell()
                         a1, a2, a3 = uc[0], uc[1], uc[2]
                         V = np.dot(a1, np.cross(a2, a3))
                         b1 = 2 * np.pi * np.cross(a2, a3) / V
                         b2 = 2 * np.pi * np.cross(a3, a1) / V
                         b3 = 2 * np.pi * np.cross(a1, a2) / V
                         B_matrix = np.array([b1, b2, b3])
                         
                         logger.info(f"Lattice Vectors:\n{uc}")
                         logger.info(f"Reciprocal Lattice Vectors:\n{B_matrix}")
                         
                         # q_vectors is a list of arrays or array of shape (N, 3)
                         # Convert to array if not
                         q_rlu = np.array(q_vectors)
                         q_cart = np.dot(q_rlu, B_matrix) # shape (N, 3) x (3, 3) -> (N, 3)
                         
                         logger.info(f"Converted Q-vectors to Cartesian (Validation: Q[0] RLU={q_rlu[0]} -> Cart={q_cart[0]})")
                         q_vectors = q_cart
                         
                     except Exception as e:
                         logger.error(f"Failed to convert Q-vectors to Cartesian: {e}")
                         # Fallback? Or raise? Assuming strict physics engine, we should probably fail or warn.
                         # But let's verify if unit_cell is valid.
                         raise e
            
             if len(q_vectors) > 0:
                 logger.info("Calculating dispersion...")
                 energies = calculator.calculate_dispersion(q_vectors)
                 
                 # Save
                 if energies:
                     # Calculate_dispersion returns List[Optional[Array]]. Convert to simpler format if possible?
                     # Standard format usually array of shape (N_q, N_modes).
                     # Ensure alignment.
                     # Filter non-None?
                     # For saving to NPZ for plotting:
                     logger.info(f"Saving dispersion results to {disp_file}...")
                     np.savez(disp_file, q_vectors=q_vectors, energies=np.array(energies, dtype=object))
                     logger.info("Dispersion calculation complete.")
                 else:
                     logger.error("Dispersion calculation failed (returned None).")
        else:
             logger.info("Dispersion calculation skipped (not needed or config set to false).")

    # 2. S(Q,w)
    if tasks.get('run_sqw_map', False):
         logger.info("--- Starting S(Q,w) Map Workflow ---")
         sqw_file = config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
         if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
         
         need_recalc_sqw = tasks.get('calculate_sqw_map_new', True) or not os.path.exists(sqw_file)
         
         if need_recalc_sqw:
             if q_vectors is None:
                 q_vectors = generate_q_path_from_config(config)
             
             if len(q_vectors) > 0:
                 logger.info("Calculating S(Q,w)...")
                 q_out, energies_out, intensities_out = calculator.calculate_sqw(q_vectors)
                 
                 if q_out:
                     logger.info(f"Saving S(Q,w) results to {sqw_file}...")
                     np.savez(sqw_file, q_vectors=q_out, energies=energies_out, intensities=intensities_out)
                     logger.info("S(Q,w) calculation complete.")
                 else:
                     logger.error("S(Q,w) calculation failed.")
         else:
             logger.info("S(Q,w) calculation skipped.")
         
    # 3. Plotting
    # Note: MagCalc doesn't have a monolithic 'plot' method exposed in the same way 
    # as the scripts usually do. The scripts call `magcalc.plotting_utils` or similar?
    # Actually, sw_KFe3J.py implements plotting locally.
    # Does MagCalc have a plotter?
    # Let's check magcalc.py content briefly or assume we need to add plotting here?
    # MagCalc.plot_dispersion_and_sqw? 
    # Checking previous file reads implies plotting logic was in sw_KFe3J.py.
    # To keep this runner simple, I'll omit plotting for now or perform a quick check.
    # Wait, the user wants "How to run the code". Ensuring plotting works is nice.
    # sw_KFe3J.py calls:
    # calculator.plot_dispersion_with_sqw(...)
    # I should add that.

    # Plotting
    plot_disp = tasks.get('plot_dispersion', False)
    plot_sqw = tasks.get('plot_sqw_map', False)
    
    if plot_disp or plot_sqw:
        logger.info("--- Plotting Results ---")
        plot_config = config.get('plotting', {})
        
        # 1. Plot Dispersion
        if plot_disp:
            disp_file = config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
            if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
            
            if os.path.exists(disp_file):
                try:
                    data = np.load(disp_file, allow_pickle=True)
                    # q_values, energies
                    q_values = data['q_vectors']
                    energies = data['energies'] # This assumes 'energies' key exists and format matches
                    
                    # Calculate path length for plotting
                    if len(q_values) > 0:
                        diffs = np.diff(q_values, axis=0)
                        dists = np.linalg.norm(diffs, axis=1)
                        path_len = np.concatenate(([0], np.cumsum(dists)))
                        x_vals = path_len
                    else:
                        x_vals = []

                    energies_list = [e for e in energies]
                    
                    plot_filename = plot_config.get('disp_plot_filename', 'disp_plot.png')
                    if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                    
                    mc.plot_dispersion_from_data(
                        q_values=x_vals,
                        energies_list=energies_list,
                        title=plot_config.get('disp_title', 'Dispersion'),
                        save_filename=plot_filename,
                        show_plot=plot_config.get('show_plot', False)
                    )
                except Exception as e:
                    logger.error(f"Failed to plot dispersion: {e}")
            else:
                logger.warning(f"Dispersion data file {disp_file} not found.")

        # 2. Plot S(Q,w)
        if plot_sqw:
            sqw_file = config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
            if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
            
            if os.path.exists(sqw_file):
                try:
                    data = np.load(sqw_file, allow_pickle=True)
                    q_vectors = data['q_vectors']
                    energies = data['energies']
                    q_vectors = data['q_vectors']
                    energies = data['energies']
                    
                    # Safely retrieve intensities
                    if 'sqw_values' in data:
                        intensities = data['sqw_values']
                    elif 'intensities' in data:
                        intensities = data['intensities']
                    else:
                         raise KeyError("No 'sqw_values' or 'intensities' key found in data file.")
                    
                    # Compute Path Length
                    if len(q_vectors) > 0:
                        # q_vectors is (N, 3)
                        diffs = np.diff(q_vectors, axis=0)
                        dists = np.linalg.norm(diffs, axis=1)
                        path_len = np.concatenate(([0], np.cumsum(dists)))
                        x_vals = path_len
                    else:
                        x_vals = np.arange(len(q_vectors))

                    plot_filename = plot_config.get('sqw_plot_filename', 'sqw_plot.png')
                    if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                    
                    # Plotting Parameters
                    title = plot_config.get('sqw_title', 'S(Q,w)')
                    ylim = plot_config.get('energy_limits_sqw', [0, 20])
                    wid = plot_config.get('broadening_width', 0.2)
                    cmap = plot_config.get('cmap', 'PuBu_r')
                    
                    logger.info(f"Plotting S(Q,w) map to {plot_filename} with broadening={wid}...")
                    
                    # --- Advanced Plotting Logic (Gaussian Broadening) ---
                    # Create Grid
                    y_min, y_max = ylim
                    dy = 0.05 # Energy resolution
                    y_grid = np.arange(y_min, y_max + dy, dy)
                    
                    # Init Matrix (Energy x Q)
                    intensity_matrix = np.zeros((len(y_grid), len(x_vals)))
                    
                    # Broadening Loop
                    for i_q in range(len(x_vals)):
                        ens = energies[i_q]
                        ints = intensities[i_q]
                        if ens is None or ints is None: continue
                        if isinstance(ens, (list, tuple)): ens = np.array(ens)
                        if isinstance(ints, (list, tuple)): ints = np.array(ints)
                        
                        # Filter NaNs
                        valid = ~np.isnan(ens) & ~np.isnan(ints)
                        ens = ens[valid]
                        ints = ints[valid]
                        
                        if len(ens) == 0: continue
                        
                        # Lorentzian Broadening (matching Manual Script)
                        for band_idx, en_val in enumerate(ens):
                            w_val = ints[band_idx]
                            denom = (y_grid - en_val)**2 + (wid/2)**2
                            lor = (1.0 / np.pi) * (wid / 2) / denom
                            intensity_matrix[:, i_q] += w_val * lor

                    # Plot
                    plt.figure(figsize=(10, 6))
                    
                    # Log Norm
                    from matplotlib.colors import LogNorm
                    
                    # Determine vmin/vmax
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
                    plt.savefig(plot_filename, dpi=150)
                    plt.close()
                    logger.info("S(Q,w) plot saved.")
                except Exception as e:
                    logger.error(f"Failed to plot S(Q,w): {e}")
            else:
                logger.warning(f"S(Q,w) data file {sqw_file} not found.")

if __name__ == "__main__":
    main()
