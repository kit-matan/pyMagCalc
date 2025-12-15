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
    
    # Check for legacy python model file
    python_model_rel = config.get('python_model_file')
    spin_model = None
    
    if python_model_rel:
        import importlib.util
        # Resolve path
        if not os.path.isabs(python_model_rel):
            python_model_path = os.path.join(config_dir, python_model_rel)
        else:
            python_model_path = python_model_rel
            
        if not os.path.exists(python_model_path):
             logger.error(f"Python model file not found: {python_model_path}")
             raise FileNotFoundError(f"Python model file not found: {python_model_path}")
             
        logger.info(f"Loading legacy python model from: {python_model_path}")
        
        try:
            # Load module dynamically
            module_name = os.path.splitext(os.path.basename(python_model_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, python_model_path)
            if spec and spec.loader:
                spin_model = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = spin_model 
                spec.loader.exec_module(spin_model)
            else:
                raise ImportError(f"Could not load spec for {python_model_path}")
                
            # Assign config to model if it expects it (GenericSpinModel does, legacy might not but harmless)
            spin_model.config = config
            
        except Exception as e:
            logger.error(f"Failed to load python model: {e}")
            raise e
    else:
        # Default: Use GenericSpinModel with config
        try:
            spin_model = GenericSpinModel(config, base_path=config_dir)
        except Exception as e:
            logger.error(f"Failed to initialize Spin Model: {e}")
            raise e
    
    final_config = config
    # Use final_config for parameters to ensure defaults
    
    # Initialize MagCalc
    calc_config = final_config.get('calculation', {})
    cache_mode = calc_config.get('cache_mode', 'r')
    cache_base = calc_config.get('cache_file_base', 'magcalc_cache')
    
    # Parameters Logic
    parameters_dict = final_config.get('parameters', {}) or final_config.get('model_params', {})
    
    # Spin Magnitude 'S' - Extract specially
    S_val = float(parameters_dict.get('S', 1.0))
    
    # Construct Hamiltonian Parameters List
    params_val = []
    param_order = final_config.get('parameter_order')
    
    # 1. Use explicit hamiltonian_params list if provided (Preferred for lists/vectors)
    if 'hamiltonian_params' in final_config:
        params_val = final_config['hamiltonian_params']
    # 2. Use explicit order if provided
    elif param_order:
        try:
             params_val = [float(parameters_dict[k]) for k in param_order]
        except KeyError as e:
             logger.error(f"Parameter in order list not found in parameters dict: {e}")
             raise e
    else:
        # Fallback: Exclude 'S' and use remaining values in insertion order?
        # A common pattern in legacy scripts.
        # But for KFe3J, dict has 'S' at start. simple values() includes it.
        # We should filter 'S' out if we are guessing.
        temp_params = {k: v for k, v in parameters_dict.items() if k != 'S'}
        params_val = list(temp_params.values())
        if python_model_rel:
            logger.warning("No 'parameter_order' specified for legacy python model. Using dictionary order (excluding 'S'). This may be incorrect.")

    logger.info(f"Model Parameters: {params_val}, S={S_val}")

    # Optimization Step
    # Legacy default checks tasks config or config root (for simple scripts)
    tasks = final_config.get('tasks', {})
    should_minimize = tasks.get('run_minimization', True)
    
    if should_minimize:
        logger.info("Minimization enabled using MagCalc.minimize_energy...")
        
        # Check for user-provided initial configuration in config
        min_config_section = final_config.get('minimization', {})
        initial_conf = min_config_section.get('initial_configuration')
        
        x0 = None
        if initial_conf:
             # Expecting list of dicts: [{atom: index/label, theta: val, phi: val}, ...]
             # Or simplified list of angles?
             # Let's support list of dicts for explicit mapping by index.
             try:
                 # Determine nspins. GenericSpinModel has atom_pos.
                 # If spin_model is a module or object with atom_pos method.
                 if hasattr(spin_model, 'atom_pos'):
                     atoms = spin_model.atom_pos()
                     nspins = len(atoms)
                     x0 = np.zeros(2 * nspins)
                     # Initialize with some default (e.g. random or zero?) 
                     # Better to random if partial info? Or zero? 
                     # MagCalc default is random if x0 is None. 
                     # If we provide x0 partially filled, we should probably fill the rest randomly or with 0.
                     # Let's fill with 0 + small noise to break symmetry if needed, or just 0.
                     x0 = np.random.uniform(0, 0.1, 2*nspins) 
                     
                     for item in initial_conf:
                         idx = item.get('atom_index')
                         if idx is None:
                             # Try label?
                             lbl = item.get('atom_label')
                             # Mapping label to index would require atom list with labels.
                             # GenericSpinModel might have access to config['crystal_structure']['atoms_uc']
                             # This is getting complicated. Let's require atom_index for now.
                             logger.warning("initial_configuration item missing 'atom_index'. Skipping.")
                             continue
                         
                         if idx >= nspins:
                             logger.warning(f"atom_index {idx} out of range for {nspins} spins.")
                             continue
                             
                         th = item.get('theta')
                         ph = item.get('phi')
                         
                         if th is not None: x0[2*idx] = float(th)
                         if ph is not None: x0[2*idx+1] = float(ph)
                         
                     logger.info("Using provided initial_configuration for minimization.")
                 else:
                     logger.warning("Cannot determine nspins from spin_model to apply initial_configuration.")
             except Exception as ex:
                 logger.error(f"Failed to parse initial_configuration: {ex}")
                 x0 = None

        # Use MagCalc for efficient minimization (skipping LSWT setup)
        # Assuming params_val for initialization is [..., H_val] which might differ from implementation_plan
        # But we pass the same params_val we built earlier.
        try:
             # Create lightweight calculator
             # Note: spin_model is the INSTANCE of GenericSpinModel (or module). 
             # If it's a module, it works. If it's an instance, checks `core.py` support.
             # core.py usually takes a module. GeneticSpinModel instance behaves like a module (has attributes).
             
             calc_min = mc.MagCalc(
                 spin_model_module=spin_model,
                 spin_magnitude=S_val,
                 hamiltonian_params=params_val,
                 cache_file_base=f"{cache_base}_min",
                 cache_mode="auto", # Cache for minimization if needed? usually not used
                 initialize=False
             )
             
             # Call minimize with optional x0
             min_res = calc_min.minimize_energy(method="L-BFGS-B", x0=x0)
             
             
             if min_res.success:
                 logger.info(f"Minimization converged. Energy: {min_res.fun:.6f}")
                 # Update the structure in spin_model
                 if hasattr(spin_model, 'set_magnetic_structure'):
                     spin_model.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                 else:
                     logger.warning("spin_model does not support 'set_magnetic_structure'. Minimization result ignored.")
                     
                 # Plot Optimized Structure if requested
                 plot_config = final_config.get('plotting', {})
                 if plot_config.get('plot_structure', False):
                     logger.info("Plotting minimized magnetic structure...")
                     try:
                         if hasattr(spin_model, 'atom_pos'):
                             atoms = spin_model.atom_pos()
                             # Determine filename
                             plot_dir = os.path.join(os.path.dirname(config_file), plot_config.get('plot_dir', '../plots')) 
                             # Config might have specific filename or we generate one
                             struct_plot_filename = plot_config.get('structure_plot_filename')
                             if not struct_plot_filename:
                                  # Fallback or default
                                  base_name = os.path.splitext(os.path.basename(config_file))[0]
                                  struct_plot_filename = os.path.join(plot_dir, f"{base_name}_structure.png")
                             else:
                                  # Ensure it is absolute or relative to config
                                  if not os.path.isabs(struct_plot_filename):
                                       struct_plot_filename = os.path.join(os.path.dirname(config_file), struct_plot_filename)
                             
                             os.makedirs(os.path.dirname(struct_plot_filename), exist_ok=True)
                             
                             mc.plot_magnetic_structure(
                                 atom_positions=atoms,
                                 spin_angles=min_res.x,
                                 title=f"Minimized Structure ({os.path.basename(config_file)})",
                                 save_filename=struct_plot_filename,
                                 show_plot=plot_config.get('show_plot', False)
                             )
                         else:
                             logger.warning("Cannot plot structure: spin_model.atom_pos() not found.")
                     except Exception as e_plot:
                         logger.error(f"Failed to plot magnetic structure: {e_plot}")

             else:
                 logger.warning(f"Minimization failed: {min_res.message}")
                 
        except Exception as e:
             logger.warning(f"Optimization attempt using MagCalc failed: {e}")
             # Fallback to internal method if it exists and failed above? 
             # Or just log error.
             
    # Initialize Main MagCalc (Heavyweight)
    logger.info("Initializing MagCalc for LSWT Calculation...")
    try:
        calculator = mc.MagCalc(
            spin_model_module=spin_model,
            spin_magnitude=S_val,
            hamiltonian_params=params_val,
            cache_file_base=cache_base,
            cache_mode=cache_mode,
        )
    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}")
        calculator = None
    
    # 1. Dispersion
    q_vectors = None
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
                 disp_res = calculator.calculate_dispersion(q_vectors_cart)
                 energies = disp_res.energies
                 
                 if energies is not None:
                     # Convert to array for saving consistency
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
             sqw_res = calculator.calculate_sqw(q_vectors_cart)
             q_out = sqw_res.q_vectors
             en_out = sqw_res.energies
             int_out = sqw_res.intensities
             
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
