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
    spin_module_name = config.get('spin_model_module')
    python_model_rel = config.get('python_model_file')
    spin_model = None
    
    if spin_module_name:
        import importlib
        try:
            spin_model = importlib.import_module(spin_module_name)
            spin_model.config = config # Inject config
            logger.info(f"Loaded spin model module: {spin_module_name}")
        except ImportError as e:
            logger.error(f"Failed to import spin_model_module '{spin_module_name}': {e}")
            raise e

    elif python_model_rel:
        import importlib.util
        if not os.path.isabs(python_model_rel):
            python_model_path = os.path.join(config_dir, python_model_rel)
        else:
            python_model_path = python_model_rel
            
        if not os.path.exists(python_model_path):
            logger.error(f"Python model file not found: {python_model_path}")
            raise FileNotFoundError(f"Python model file not found: {python_model_path}")
             
        logger.info(f"Loading legacy python model from: {python_model_path}")
        
        try:
            module_name = os.path.splitext(os.path.basename(python_model_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, python_model_path)
            if spec and spec.loader:
                spin_model = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = spin_model 
                spec.loader.exec_module(spin_model)
            else:
                raise ImportError(f"Could not load spec for {python_model_path}")
                
            spin_model.config = config
        except Exception as e:
            logger.error(f"Failed to load python model: {e}")
            raise e
    else:
        try:
            spin_model = GenericSpinModel(config, base_path=config_dir)
        except Exception as e:
            logger.error(f"Failed to initialize Spin Model: {e}")
            raise e
    
    final_config = config
    
    # Initialize MagCalc
    calc_config = final_config.get('calculation', {})
    cache_mode = calc_config.get('cache_mode', 'none')
    cache_base = calc_config.get('cache_file_base', 'magcalc_cache')
    
    # Parameters Logic
    parameters_dict = final_config.get('parameters', {}) or final_config.get('model_params', {})
    
    # Try to get S from parameters, otherwise check first atom's spin_S
    if 'S' in parameters_dict:
        S_val = float(parameters_dict['S'])
    else:
        # Fallback to first atom's spin
        try:
            atoms = final_config.get('crystal_structure', {}).get('atoms_uc', [])
            if atoms:
                S_val = float(atoms[0].get('spin_S', 1.0))
            else:
                S_val = 1.0
        except (ValueError, TypeError, IndexError):
            S_val = 1.0
    
    params_val = []
    param_order = final_config.get('parameter_order')
    
    if 'hamiltonian_params' in final_config:
        params_val = final_config['hamiltonian_params']
    elif param_order:
        try:
            params_val = []
            for k in param_order:
                val = parameters_dict[k]
                if isinstance(val, (list, tuple, np.ndarray)):
                    params_val.append([float(x) for x in val])
                else:
                    params_val.append(float(val))
        except KeyError as e:
            logger.error(f"Parameter in order list not found in parameters dict: {e}")
            raise e
    else:
        temp_params = {k: v for k, v in parameters_dict.items() if k != 'S'}
        params_val = list(temp_params.values())
        if python_model_rel:
            logger.warning("No 'parameter_order' specified for legacy python model. Using dictionary order (excluding 'S'). This may be incorrect.")

    logger.info(f"Model Parameters: {params_val}, S={S_val}")

    tasks = final_config.get('tasks', {})
    # 1. Minimization
    do_minimization = tasks.get('minimization', False)
    # Check legacy
    if 'run_minimization' in tasks: do_minimization = tasks['run_minimization']
    
    if do_minimization:
        # Check if python model provides minimization
        if spin_model and hasattr(spin_model, 'minimize'):
            logger.info("Minimization handled by custom spin model 'minimize' method.")
            try:
                min_res = spin_model.minimize(config) # Pass full config for flexibility
                if min_res.success:
                    logger.info(f"Custom minimization converged. Energy: {min_res.fun:.6f}")
                    # Assume custom minimize sets the magnetic structure internally or returns it
                    # If it returns x, we can set it here:
                    if hasattr(spin_model, 'set_magnetic_structure') and hasattr(min_res, 'x'):
                        spin_model.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                else:
                    logger.warning(f"Custom minimization failed: {min_res.message}")
            except Exception as e:
                logger.error(f"Custom spin model minimization failed: {e}")
        else:
            logger.info("Minimization enabled using MagCalc.minimize_energy...")
            x0 = None
            if hasattr(spin_model, 'generate_magnetic_structure'):
                try:
                    thetas, phis = spin_model.generate_magnetic_structure()
                    if thetas is not None and phis is not None:
                        nspins = len(thetas)
                        x0 = np.zeros(2 * nspins)
                        x0[0::2] = thetas
                        x0[1::2] = phis
                        logger.info("Generated initial magnetic structure from 'magnetic_structure' config.")
                except Exception as e:
                    logger.error(f"Failed to generate magnetic structure: {e}")

            if x0 is None:
                min_config_section = final_config.get('minimization', {})
                initial_conf = min_config_section.get('initial_configuration')
                
                if initial_conf:
                    try:
                        if hasattr(spin_model, 'atom_pos'):
                            atoms = spin_model.atom_pos()
                            nspins = len(atoms)
                            x0 = np.random.uniform(0, 0.1, 2*nspins) 
                            
                            for item in initial_conf:
                                idx = item.get('atom_index')
                                if idx is None: continue
                                if idx >= nspins: continue
                                th = item.get('theta')
                                ph = item.get('phi')
                                if th is not None: x0[2*idx] = float(th)
                                if ph is not None: x0[2*idx+1] = float(ph)
                            logger.info("Using provided initial_configuration for minimization.")
                    except Exception as ex:
                        logger.error(f"Failed to parse initial_configuration: {ex}")
                        x0 = None

            try:
                calc_min = mc.MagCalc(
                    spin_model_module=spin_model,
                    spin_magnitude=S_val,
                    hamiltonian_params=params_val,
                    cache_file_base=f"{cache_base}_min",
                    cache_mode="none",
                    initialize=False
                )
                
                min_config_section = final_config.get('minimization', {})
                num_starts = min_config_section.get('num_starts', 1)
                n_workers = min_config_section.get('n_workers', 1)
                early_stopping = min_config_section.get('early_stopping', 0)
                
                min_res = calc_min.minimize_energy(
                    method="L-BFGS-B", 
                    x0=x0, 
                    num_starts=num_starts, 
                    n_workers=n_workers, 
                    early_stopping=early_stopping
                )
                
                if min_res.success:
                    logger.info(f"Minimization converged. Energy: {min_res.fun:.6f}")
                    if hasattr(spin_model, 'set_magnetic_structure'):
                        spin_model.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                    
                    plot_config = final_config.get('plotting', {})
                    if plot_config.get('plot_structure', False):
                        logger.info("Plotting minimized magnetic structure...")
                        try:
                            if hasattr(spin_model, 'atom_pos'):
                                atoms = spin_model.atom_pos()
                                plot_dir = os.path.join(os.path.dirname(config_file), plot_config.get('plot_dir', '../plots')) 
                                struct_plot_filename = plot_config.get('structure_plot_filename')
                                if not struct_plot_filename:
                                    base_name = os.path.splitext(os.path.basename(config_file))[0]
                                    struct_plot_filename = os.path.join(plot_dir, f"{base_name}_structure.png")
                                else:
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
                        except Exception as e_plot:
                            logger.error(f"Failed to plot magnetic structure: {e_plot}")
                else:
                    logger.warning(f"Minimization failed: {min_res.message}")
            except Exception as e:
                logger.warning(f"Optimization attempt using MagCalc failed: {e}")

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
    
    # Store calculated data for plotting if not saved to file
    memory_cache = {
        'dispersion': None,
        'sqw': None
    }
    
    save_data_flag = final_config.get('output', {}).get('save_data', True)

    # 2. Dispersion
    q_vectors = None
    do_dispersion = tasks.get('dispersion', False)
    if 'run_dispersion' in tasks: do_dispersion = tasks['run_dispersion']
    
    if do_dispersion:
        disp_file = final_config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
        if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
        
        need_recalc = True # If dispersion=True, we generally want to run unless cached logic is added back
        # Legacy key check
        if 'calculate_dispersion_new' in tasks: need_recalc = tasks['calculate_dispersion_new']
        if not os.path.exists(disp_file) and save_data_flag: need_recalc = True
        
        if need_recalc:
            if q_vectors is None:
                q_vectors = generate_q_path_from_config(final_config)
            
            if len(q_vectors) > 0:
                uc = spin_model.unit_cell()
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
                    try:
                        energies_arr = np.array(energies)
                    except:
                        energies_arr = np.array(energies, dtype=object)

                    memory_cache['dispersion'] = {
                        'q_vectors': q_vectors_cart,
                        'energies': energies_arr
                    }

                    if save_data_flag:
                        os.makedirs(os.path.dirname(disp_file), exist_ok=True)
                        np.savez(disp_file, q_vectors=q_vectors_cart, energies=energies_arr)
                        logger.info(f"Dispersion saved to {disp_file}")

                    if tasks.get('export_csv', False):
                        disp_csv = final_config.get('output', {}).get('disp_csv_filename', 'disp_data.csv')
                        if not os.path.isabs(disp_csv): disp_csv = os.path.join(config_dir, disp_csv)
                        
                        logger.info(f"Exporting dispersion to CSV: {disp_csv}")
                        # qx, qy, qz, en1, en2, ...
                        header = "qx,qy,qz," + ",".join([f"en{i}" for i in range(energies_arr.shape[1])])
                        with open(disp_csv, 'w') as f:
                            f.write(header + "\n")
                            for i in range(len(q_vectors_cart)):
                                q = q_vectors_cart[i]
                                en = energies_arr[i]
                                line = f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f}," + ",".join([f"{e:.6f}" for e in en])
                                f.write(line + "\n")
                
            else:
                logger.warning("No Q-vectors.")
    
    # 3. S(Q,w) Map
    do_sqw = tasks.get('sqw_map', False)
    if 'run_sqw_map' in tasks: do_sqw = tasks['run_sqw_map']
    
    if do_sqw:
        sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
        if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
        
        need_recalc = True
        if 'calculate_sqw_map_new' in tasks: need_recalc = tasks['calculate_sqw_map_new']
        if not os.path.exists(sqw_file) and save_data_flag: need_recalc = True
        
        if need_recalc:
            if q_vectors is None:
                q_vectors = generate_q_path_from_config(final_config)
            
            if q_vectors is not None and len(q_vectors) > 0:
                uc = spin_model.unit_cell()
                a1, a2, a3 = uc[0], uc[1], uc[2]
                V = np.dot(a1, np.cross(a2, a3))
                B_matrix = np.array([
                    2 * np.pi * np.cross(a2, a3) / V,
                    2 * np.pi * np.cross(a3, a1) / V,
                    2 * np.pi * np.cross(a1, a2) / V
                ])
                q_vectors_cart = np.dot(q_vectors, B_matrix)

                logger.info("Calculating S(Q,w)...")
                sqw_res = calculator.calculate_sqw(q_vectors_cart)
                q_out = sqw_res.q_vectors
                en_out = sqw_res.energies
                int_out = sqw_res.intensities
                
                memory_cache['sqw'] = {
                    'q_vectors': q_out,
                    'energies': en_out,
                    'intensities': int_out
                }

                if save_data_flag:
                    os.makedirs(os.path.dirname(sqw_file), exist_ok=True)
                    np.savez(sqw_file, q_vectors=q_out, energies=en_out, intensities=int_out)
                    logger.info(f"S(Q,w) saved to {sqw_file}")

                if tasks.get('export_csv', False):
                    sqw_csv = final_config.get('output', {}).get('sqw_csv_filename', 'sqw_data.csv')
                    if not os.path.isabs(sqw_csv): sqw_csv = os.path.join(config_dir, sqw_csv)
                    
                    logger.info(f"Exporting S(Q,w) to CSV: {sqw_csv}")
                    # qx, qy, qz, mode, energy, intensity
                    header = "qx,qy,qz,mode,energy,intensity"
                    with open(sqw_csv, 'w') as f:
                        f.write(header + "\n")
                        for i in range(len(q_out)):
                            q = q_out[i]
                            # en_out and int_out are [nq, nmodes]
                            for m in range(en_out.shape[1]):
                                line = f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{m},{en_out[i,m]:.6f},{int_out[i,m]:.6f}"
                                f.write(line + "\n")

    # 3. Plotting
    plot_config = final_config.get('plotting', {})
    
    # Logic: If 'dispersion' task is ON, and 'save_plot' is ON, then plot.
    # Also support legacy 'plot_dispersion' key if present.
    should_plot_disp = do_dispersion and plot_config.get('save_plot', True)
    if 'plot_dispersion' in tasks: should_plot_disp = tasks['plot_dispersion'] and plot_config.get('save_plot', True)

    if should_plot_disp:
        # Prefer memory cache
        if memory_cache['dispersion'] is not None:
             data = memory_cache['dispersion']
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
        else:
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

    should_plot_sqw = do_sqw and plot_config.get('save_plot', True)
    if 'plot_sqw_map' in tasks: should_plot_sqw = tasks['plot_sqw_map'] and plot_config.get('save_plot', True)

    if should_plot_sqw:
        # Prefer memory cache
        if memory_cache['sqw'] is not None:
            data = memory_cache['sqw']
            plot_filename = plot_config.get('sqw_plot_filename', 'sqw_plot.png')
            if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
            
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
        else:
            sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
            if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
            
            if os.path.exists(sqw_file):
                data = np.load(sqw_file, allow_pickle=True)
                plot_filename = plot_config.get('sqw_plot_filename', 'sqw_plot.png')
                if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                
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
