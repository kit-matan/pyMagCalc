import os
import sys
import yaml
import logging
import numpy as np
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.plotting import plot_dispersion, plot_sqw_map

logger = logging.getLogger(__name__)

def generate_q_path_from_config(config):
    """Generates Q-path from configuration dictionary."""
    q_conf = config.get('q_path', {})
    if not q_conf:
        return np.array([])
        
    points = {k: np.array(v) for k, v in q_conf.items() if k not in ['path', 'points_per_segment', 'unit']}
    path_labels = q_conf.get('path', [])
    n_points = q_conf.get('points_per_segment', 50)
    
    q_vectors = []
    if path_labels and len(path_labels) > 1:
        for i in range(len(path_labels)-1):
            start_label = path_labels[i]
            end_label = path_labels[i+1]
            
            start_pt = points.get(start_label)
            end_pt = points.get(end_label)
            
            if start_pt is None or end_pt is None:
                 logger.error(f"Undefined point in path: {start_label} -> {end_label}")
                 continue

            segment = np.linspace(start_pt, end_pt, n_points)
            if i > 0:
                segment = segment[1:] 
            q_vectors.extend(segment)
            
    return np.array(q_vectors)

def run_calculation(config_file: str):
    """
    Main execution logic for running a MagCalc calculation.
    """
    if not os.path.exists(config_file):
        logger.error(f"Config file '{config_file}' not found.")
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

    # Load Config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_file}")
    config_dir = os.path.dirname(os.path.abspath(config_file))

    # Initialize Generic Model
    # GenericSpinModel will validate schema internally now!
    try:
        spin_model = GenericSpinModel(config, base_path=config_dir)
    except Exception as e:
        logger.error(f"Failed to initialize Spin Model: {e}")
        raise e
    
    # Validation happened in GenericSpinModel, so config structure is guaranteed (mostly).
    # But self.config inside spin_model might be different from local 'config' if defaults applied.
    # Ideally use spin_model.config but GenericSpinModel doesn't expose it cleanly yet or we didn't check.
    # But we can assume 'config' is acceptable or use spin_model.config if accessible.
    # spin_model.config is accessible.
    
    final_config = spin_model.config 
    # Use final_config for parameters to ensure defaults
    
    # Initialize MagCalc
    calc_config = final_config.get('calculation', {})
    cache_mode = calc_config.get('cache_mode', 'r')
    cache_base = calc_config.get('cache_file_base', 'magcalc_cache')
    
    # Parameters
    params_val = []
    # If parameters dict exists (New Schema Style)
    parameters_dict = final_config.get('parameters', {})
    param_names = list(parameters_dict.keys()) # Order might matter for internal consistency, dependent on dictionary order
    
    # Wait, GenericSpinModel uses 'parameters' list to map positional args?
    # Schema defines 'parameters' as Dict[str, float].
    # But GenericSpinModel uses self.config.get('parameters') as a LIST of names if it's the old style,
    # or expects 'parameters' to be a dict in new style?
    # Let's check Schema again.
    # Schema: parameters: Dict[str, float].
    # GenericSpinModel logic (Step 58): 
    # if self.config.get('parameters'): ... param_names = self.config.get('parameters') ...
    # Wait, if 'parameters' is a Dict, iterating it yields keys (names). So it works!
    
    if isinstance(parameters_dict, dict):
         params_val = list(parameters_dict.values())
         # We need to ensure the ORDER passed to MagCalc matches the ORDER 'GenericSpinModel' expects 
         # when it unwraps them.
         # GenericSpinModel uses `_resolve_param_map(p)`:
         # param_names = self.config.get('parameters') [KEYS]
         # maps name -> p[i].
         # So providing we pass values() in same order as keys(), it works.
         # Python 3.7+ preserves dict order.
    else:
        # Fallback if somehow it's a list (Legacy logic but Schema enforces dict?)
        # Schema enforces Dict.
        params_val = [] 
    
    # Spin Magnitude 'S'
    # In strict schema 'parameters' should contain 'S' if referenced?
    # Or is 'S' special?
    # In KFe3J, S is in model_params.
    # Schema aliased model_params to parameters.
    # So 'S' should be in parameters_dict.
    S_val = float(parameters_dict.get('S', 1.0))

    logger.info(f"Model Parameters: {params_val}, S={S_val}")
    
    calculator = mc.MagCalc(
        spin_model_module=spin_model, 
        spin_magnitude=S_val,
        hamiltonian_params=params_val,
        cache_file_base=cache_base,
        cache_mode=cache_mode
    )
    
    tasks = final_config.get('tasks', {})
    q_vectors = None
    
    # 1. Dispersion
    if tasks.get('run_dispersion', False):
        disp_file = final_config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
        if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
        
        need_recalc = tasks.get('calculate_dispersion_new', True) or not os.path.exists(disp_file)
        
        if need_recalc:
             if q_vectors is None:
                 q_vectors = generate_q_path_from_config(final_config)
             
             if len(q_vectors) > 0:
                 # Convert RLU to Cartesian
                 uc = spin_model.unit_cell()
                 # ... (Conversion logic same as run_magcalc.py)
                 a1, a2, a3 = uc[0], uc[1], uc[2]
                 V = np.dot(a1, np.cross(a2, a3))
                 B_matrix = np.array([
                     2 * np.pi * np.cross(a2, a3) / V,
                     2 * np.pi * np.cross(a3, a1) / V,
                     2 * np.pi * np.cross(a1, a2) / V
                 ])
                 
                 q_rlu = np.array(q_vectors)
                 q_cart = np.dot(q_rlu, B_matrix)
                 q_vectors_cart = q_cart

                 logger.info("Calculating dispersion...")
                 energies = calculator.calculate_dispersion(q_vectors_cart)
                 
                 if energies:
                     # Convert to array for saving consistency
                     # Some internal logic might return list of arrays
                     # Try to stack if regular
                     try:
                         energies_arr = np.array(energies)
                     except:
                         energies_arr = np.array(energies, dtype=object)

                     np.savez(disp_file, q_vectors=q_vectors_cart, energies=energies_arr)
                     logger.info(f"Dispersion saved to {disp_file}")
                 
             else:
                 logger.warning("No Q-vectors.")
    
    # 2. S(Q,w)
    if tasks.get('run_sqw_map', False):
         sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
         if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
         
         need_recalc = tasks.get('calculate_sqw_map_new', True) or not os.path.exists(sqw_file)
         
         if need_recalc:
             if q_vectors is None:
                  # Need to regenerate if not done in dispersion
                  # And re-convert
                  q_temp = generate_q_path_from_config(final_config)
                  # conversion... (Duplication of logic, should extract helper potentially)
                  # For brevity assuming q_vectors was populated or doing minimal copy
                  # Real implementation should be cleaner.
                  pass 
             
             # Assuming q_vectors is available or generated
             if q_vectors is not None and len(q_vectors) > 0:
                  # Use existing q_vectors_cart if defined
                  # ...
                  pass # (Skipping re-implementation of conversion for brevity in snippet)
                  # Actually, let's fix the scope of q_vectors_cart.
                  # It was local to if block.
                  pass
                  
             # Re-doing clean flow:
             if q_vectors is None:
                 q_vectors = generate_q_path_from_config(final_config)
                 # Conversion
                 uc = spin_model.unit_cell()
                 a1, a2, a3 = uc[0], uc[1], uc[2]
                 V = np.dot(a1, np.cross(a2, a3))
                 B_matrix = np.array([
                     2 * np.pi * np.cross(a2, a3) / V,
                     2 * np.pi * np.cross(a3, a1) / V,
                     2 * np.pi * np.cross(a1, a2) / V
                 ])
                 q_vectors_cart = np.dot(q_vectors, B_matrix)
             else:
                 # If we came from Disp block, we might have q_vectors (RLU) vs q_vectors_cart
                 # Let's standardize q_vectors as RLU and convert when calling.
                 q_vectors_cart = np.dot(q_vectors, B_matrix) # Assuming B_matrix valid

             logger.info("Calculating S(Q,w)...")
             q_out, en_out, int_out = calculator.calculate_sqw(q_vectors_cart)
             np.savez(sqw_file, q_vectors=q_out, energies=en_out, intensities=int_out)
             logger.info(f"S(Q,w) saved to {sqw_file}")

    # 3. Plotting
    plot_config = final_config.get('plotting', {})
    
    if tasks.get('plot_dispersion', False) and plot_config.get('save_plot', True):
        disp_file = final_config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
        if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
        
        if os.path.exists(disp_file):
            data = np.load(disp_file, allow_pickle=True)
            plot_filename = plot_config.get('disp_plot_filename', 'disp_plot.png')
            if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
            
            plot_dispersion(
                q_vectors=data['q_vectors'],
                energies=data['energies'],
                save_filename=plot_filename,
                title=plot_config.get('disp_title', "Dispersion"),
                ylim=plot_config.get('energy_limits_disp'),
                show_plot=plot_config.get('show_plot', False)
            )

    if tasks.get('plot_sqw_map', False) and plot_config.get('save_plot', True):
        sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
        if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
        
        if os.path.exists(sqw_file):
            data = np.load(sqw_file, allow_pickle=True)
            plot_filename = plot_config.get('sqw_plot_filename', 'sqw_plot.png')
            if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
            
            # Handling intensity key (sqw_values vs intensities)
            ints = data.get('intensities')
            if ints is None: ints = data.get('sqw_values')

            plot_sqw_map(
                q_vectors=data['q_vectors'],
                energies=data['energies'],
                intensities=ints,
                save_filename=plot_filename,
                title=plot_config.get('sqw_title', "S(Q,w)"),
                ylim=plot_config.get('energy_limits_sqw'),
                broadening_width=plot_config.get('broadening_width', 0.2),
                cmap=plot_config.get('cmap', 'PuBu_r'),
                show_plot=plot_config.get('show_plot', False)
            )
