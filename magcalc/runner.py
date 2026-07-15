import os
import sys
import yaml
import logging
import numpy as np
import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.plotting import plot_dispersion, plot_sqw_map, plot_energy_cuts

logger = logging.getLogger(__name__)

def _grid_axis_coords(origin, vec, fractions):
    """Human-friendly coordinates and label for one axis of a 2-D q grid.

    When vec lies along a single RLU axis, the coordinate is that RLU
    component (e.g. H running origin_H .. origin_H + vec_H) labelled H/K/L;
    otherwise the raw fraction 0..1 along vec is used and the label spells
    out the vector.
    """
    vec = np.asarray(vec, dtype=float)
    origin = np.asarray(origin, dtype=float)
    nonzero = np.nonzero(np.abs(vec) > 1e-12)[0]
    if len(nonzero) == 1:
        i = int(nonzero[0])
        return origin[i] + fractions * vec[i], f"{'HKL'[i]} (r.l.u.)"
    return fractions, f"fraction along {vec.tolist()} (r.l.u.)"


def compute_b_matrix(spin_model):
    """Compute the reciprocal lattice B-matrix from the spin model's unit cell.

    When a magnetic supercell is active, the CHEMICAL cell is used so that
    q_path entries stay in chemical-cell RLU (SpinW/Sunny convention) and the
    supercell merely folds the bands.
    """
    if getattr(spin_model, 'supercell_dims', [1, 1, 1]) != [1, 1, 1] and \
            hasattr(spin_model, 'chemical_unit_cell'):
        uc = spin_model.chemical_unit_cell()
    else:
        uc = spin_model.unit_cell()
    a1, a2, a3 = uc[0], uc[1], uc[2]
    V = np.dot(a1, np.cross(a2, a3))
    return np.array([
        2 * np.pi * np.cross(a2, a3) / V,
        2 * np.pi * np.cross(a3, a1) / V,
        2 * np.pi * np.cross(a1, a2) / V
    ])

def _safe_makedirs(filepath):
    """Create parent directories for filepath if they exist."""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

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

def _resolve_plot_structure_flag(tasks, plot_config):
    """Priority: tasks['plot_structure'] -> tasks['run_plotting'] -> plotting['plot_structure']."""
    flag = tasks.get('plot_structure')
    if flag is None:
        flag = tasks.get('run_plotting')
    if flag is None:
        flag = plot_config.get('plot_structure', False)
    return flag


def _plot_structure_outputs(spin_model, spin_angles, energy, final_config, config_file, title):
    """Render the magnetic structure (PNG + JSON for the GUI's interactive
    viewer) from interleaved spin angles [th0, ph0, th1, ph1, ...].

    Used both for minimized structures and for manual structures supplied via
    the 'magnetic_structure' config with minimization skipped."""
    plot_config = final_config.get('plotting', {})
    try:
        if not hasattr(spin_model, 'atom_pos'):
            return
        atoms = spin_model.atom_pos()
        plot_dir = os.path.join(os.path.dirname(config_file), plot_config.get('plot_dir', '../plots'))
        struct_plot_filename = plot_config.get('structure_plot_filename')
        if not struct_plot_filename:
            base_name = os.path.splitext(os.path.basename(config_file))[0]
            struct_plot_filename = os.path.join(plot_dir, f"{base_name}_structure.png")
        elif not os.path.isabs(struct_plot_filename):
            struct_plot_filename = os.path.join(os.path.dirname(config_file), struct_plot_filename)

        os.makedirs(os.path.dirname(struct_plot_filename), exist_ok=True)

        mc.plot_magnetic_structure(
            atom_positions=atoms,
            spin_angles=spin_angles,
            title=title,
            save_filename=struct_plot_filename,
            show_plot=plot_config.get('show_plot', False)
        )

        # JSON for the interactive 3D viewer
        try:
            json_filename = struct_plot_filename.replace('.png', '.json')
            vectors = []
            for i in range(len(atoms)):
                th = spin_angles[2 * i]
                ph = spin_angles[2 * i + 1]
                vectors.append([float(np.sin(th) * np.cos(ph)),
                                float(np.sin(th) * np.sin(ph)),
                                float(np.cos(th))])

            import json
            structure_data = {
                "atoms": atoms.tolist(),
                "vectors": vectors
            }
            if energy is not None:
                structure_data["energy"] = float(energy)

            with open(json_filename, 'w') as f:
                json.dump(structure_data, f, indent=2)
            logger.info(f"Saved interactive structure data to {json_filename}")
        except Exception as e_json:
            logger.error(f"Failed to save structure JSON: {e_json}")
    except Exception as e_plot:
        logger.error(f"Failed to plot magnetic structure: {e_plot}")



def _advise_sun_mode(spin_model):
    """Warn when a model has single-ion physics that dipole LSWT structurally CANNOT see.

    Dipole LSWT expands each spin as ONE boson, so it has no transitions between an ion's
    local crystal-field levels. Whenever S >= 1 AND there is an on-site anisotropy, those
    single-ion (multipolar) excitations exist and are simply ABSENT from the spectrum --
    silently. FeI2's bound state is the textbook case. Nothing about the output looks
    wrong; entire bands are just missing.

    Only fires when it can actually matter (S >= 1 and an anisotropy present), so it is
    not noise: an S=1/2 model has no multipolar levels, and without anisotropy the
    multipolar modes carry no weight.
    """
    try:
        mags = spin_model.spin_magnitudes()
        inters = spin_model.interactions_config
    except Exception:
        return
    if not mags or max(float(m) for m in mags) < 1.0:
        return
    kinds = {'sia', 'sia_matrix', 'anisotropy_matrix', 'stevens'}
    if not any((i.get('type') in kinds) for i in inters if isinstance(i, dict)):
        return
    logger.warning(
        "This model has S >= 1 AND a single-ion anisotropy, so it has single-ion "
        "(multipolar) excitations -- and dipole LSWT structurally cannot represent them: "
        "those bands will be MISSING from the spectrum, with nothing to indicate it. "
        "If they matter (FeI2's bound state is the classic example), use "
        "`calculation: {mode: SUN}`. See CLAUDE.md 5c."
    )


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

    # mCIF: an experimentally-determined magnetic structure (magnetic space group +
    # moments). `from_mcif: <file>` fills crystal_structure (lattice + magnetic-cell
    # atoms) and magnetic_structure (per-site directions) from the mCIF; the user still
    # supplies `interactions`, `parameters`, `tasks`. An explicit crystal_structure /
    # magnetic_structure in the config takes precedence over the mCIF-derived one.
    _mcif = config.get('from_mcif')
    if _mcif:
        from magcalc.mcif import mcif_to_config_fragment
        mpath = _mcif if os.path.isabs(_mcif) else os.path.join(config_dir, _mcif)
        _mcfg = config.get('mcif', {}) or {}
        frag = mcif_to_config_fragment(mpath, spin_S=_mcfg.get('spin_S', 1.0),
                                       ion=_mcfg.get('ion'))
        logger.info(f"Loaded magnetic structure from mCIF {mpath}: "
                    f"{len(frag['crystal_structure']['atoms_uc'])} sites.")
        frag['crystal_structure'].update(config.get('crystal_structure') or {})
        config['crystal_structure'] = frag['crystal_structure']
        config.setdefault('magnetic_structure', frag['magnetic_structure'])

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
    
    final_config = spin_model.config
    
    # Initialize MagCalc
    calc_config = final_config.get('calculation', {})
    # Default to 'auto' for better performance in iterative GUI/CLI usage
    cache_mode = calc_config.get('cache_mode', 'auto')
    cache_base = calc_config.get('cache_file_base', 'magcalc_run_cache')
    # Compute backend: 'numpy' (default) or 'fortran' (external fMagCalc, with
    # automatic fallback to NumPy if unavailable). Applies to S(Q,w) and powder.
    backend = calc_config.get('backend', 'numpy')

    # Sample environment / measurement model, applied to S(Q,w), powder and
    # energy-cut intensities:
    #   temperature: Kelvin -> Bose thermal prefactor per mode.
    #   domains: twin averaging ({axis, n_fold} or explicit list; see
    #       core._parse_domains). Dispersion stays single-domain.
    #   cross_section: 'perp' (default) | 'trace' | component ('xx', 'xy', ...).
    temperature = calc_config.get('temperature')
    domains = calc_config.get('domains')
    cross_section = calc_config.get('cross_section', 'perp')
    
    # Parameters Logic
    parameters_dict = final_config.get('parameters')
    if parameters_dict is None:
        parameters_dict = final_config.get('model_params', {})
    
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
    
    if S_val <= 0:
        logger.error(f"Invalid spin magnitude S={S_val}. S must be positive.")
        raise ValueError(f"Spin magnitude must be positive (S={S_val}). Please check your configuration.")
    
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
    q_vectors = None
    do_minimization = tasks.get('minimization', False)
    
    min_config_section_early = final_config.get('minimization', {}) or {}
    if do_minimization and min_config_section_early.get('optimize_k') \
            and isinstance(spin_model, GenericSpinModel):
        # Spiral (single-k) ground-state search: optimize k, the spin
        # directions, and optionally the rotation axis (Sunny
        # minimize_spiral_energy! / SpinW optmagstr analogue). Commits the
        # optimum to the model, so the MagCalc built below bakes in the
        # optimized k; the plain angle-only minimization is skipped.
        from magcalc import spiral_opt
        logger.info("Minimization with optimize_k: running spiral (single-k) optimizer...")
        try:
            sp_res = spiral_opt.optimize_spiral(
                spin_model, params_val, min_config_section_early, S_val=S_val)
            logger.info(sp_res.message)

            out_cfg = final_config.get('output', {})
            opt_file = out_cfg.get('optimized_structure_filename', 'optimized_structure.yaml')
            if not os.path.isabs(opt_file):
                opt_file = os.path.join(config_dir, opt_file)
            try:
                _safe_makedirs(opt_file)
                with open(opt_file, 'w') as f:
                    yaml.safe_dump({'magnetic_structure': {
                        'enabled': True,
                        'type': 'single_k',
                        'k': [float(c) for c in sp_res.k_rlu],
                        'axis': [float(c) for c in sp_res.axis],
                        'local_directions': [[float(c) for c in d]
                                             for d in spin_model.mag_struct_cfg['local_directions']],
                    }, 'energy_per_site': float(sp_res.energy_per_site)}, f,
                        default_flow_style=None, sort_keys=False)
                logger.info(f"Optimized single-k structure written to {opt_file}")
            except Exception as e:
                logger.warning(f"Could not write optimized structure file: {e}")

            if _resolve_plot_structure_flag(tasks, final_config.get('plotting', {})):
                spin_angles = np.ravel(np.column_stack([sp_res.thetas, sp_res.phis]))
                _plot_structure_outputs(
                    spin_model, spin_angles, sp_res.energy, final_config, config_file,
                    title=f"Optimized spiral k={np.round(sp_res.k_rlu, 5).tolist()} "
                          f"({os.path.basename(config_file)})"
                )
        except Exception as e:
            logger.error(f"Spiral (optimize_k) minimization failed: {e}")
            raise
    elif do_minimization:
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
                method = min_config_section.get('method', 'TNC')

                # early_stopping = "stop after N starts hit the same minimum". A flat
                # default of 10 is too permissive for big frustrated cells: SW20 in
                # field (16 sites = 32 angles) settled in a LOCAL minimum at 10 and
                # produced imaginary magnons everywhere. Scale it with the number of
                # free angles unless the user pins it.
                n_sites = len(spin_model.atom_pos()) if hasattr(spin_model, 'atom_pos') else 1
                default_es = max(10, 2 * n_sites)
                early_stopping = min_config_section.get('early_stopping', default_es)
                if 'early_stopping' not in min_config_section and default_es > 10:
                    logger.info(
                        f"minimization.early_stopping defaulted to {default_es} "
                        f"(2 x {n_sites} sites); pin it explicitly to override.")
                if num_starts > 1 and num_starts < early_stopping:
                    logger.warning(
                        f"minimization: num_starts ({num_starts}) < early_stopping "
                        f"({early_stopping}), so early stopping can never trigger and the "
                        f"minimum may be a local one. Raise num_starts.")

                # Extract extra kwargs for the minimizer
                excluded_keys = {'num_starts', 'n_workers', 'early_stopping', 'method', 'initial_configuration', 'enabled'}
                min_kwargs = {k: v for k, v in min_config_section.items() if k not in excluded_keys}
                
                min_res = calc_min.minimize_energy(
                    method=method, 
                    x0=x0, 
                    num_starts=num_starts, 
                    n_workers=n_workers, 
                    early_stopping=early_stopping,
                    **min_kwargs
                )
                
                if min_res.success:
                    logger.info(f"Global minimization complete (method={method}).")
                else:
                    logger.warning(f"Minimization failed to fully converge: {min_res.message}")
                    
                if hasattr(min_res, 'x') and min_res.x is not None:
                    logger.info(f"Best energy: {min_res.fun:.6f} meV ({getattr(min_res, 'global_message', '')})")
                    if hasattr(spin_model, 'set_magnetic_structure'):
                        spin_model.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                    
                    plot_config = final_config.get('plotting', {})
                    if _resolve_plot_structure_flag(tasks, plot_config):
                        logger.info("Plotting minimized magnetic structure...")
                        _plot_structure_outputs(
                            spin_model, min_res.x, min_res.fun, final_config, config_file,
                            title=f"Minimized Structure ({os.path.basename(config_file)})"
                        )
            except Exception as e:
                logger.warning(f"Optimization attempt using MagCalc failed: {e}")

    # Apply a manual magnetic structure when minimization is NOT run. This lets a
    # structure obtained from a previous energy minimization (e.g. imported into
    # the GUI's Manual Structure tab) drive the LSWT calculation directly, without
    # re-minimizing every run. Skipped if minimization already set the structure.
    elif spin_model is not None and hasattr(spin_model, 'generate_magnetic_structure'):
        ms_cfg = final_config.get('magnetic_structure') or {}
        if ms_cfg.get('type') and ms_cfg.get('enabled', True):
            try:
                thetas, phis = spin_model.generate_magnetic_structure()
                if thetas is not None and phis is not None and \
                        hasattr(spin_model, 'set_magnetic_structure'):
                    spin_model.set_magnetic_structure(thetas, phis)
                    logger.info(
                        "Applied manual magnetic structure from config "
                        "(minimization skipped)."
                    )
                    # Render the supplied structure so the GUI's Magnetic
                    # Structure panel shows it even without minimization.
                    if _resolve_plot_structure_flag(tasks, final_config.get('plotting', {})):
                        logger.info("Plotting manual magnetic structure...")
                        spin_angles = np.ravel(np.column_stack([thetas, phis]))
                        _plot_structure_outputs(
                            spin_model, spin_angles, None, final_config, config_file,
                            title=f"Manual Structure ({os.path.basename(config_file)})"
                        )
            except Exception as e:
                logger.warning(f"Failed to apply manual magnetic structure: {e}")

    # Initialize Main MagCalc (Heavyweight)
    # Engine: 'dipole' (default) or 'SUN'. SU(N) carries an N-level local Hilbert space
    # per site, so it represents SINGLE-ION (multipolar) excitations -- FeI2's bound
    # state, for instance -- which dipole LSWT structurally cannot. Everything else
    # (structure, symmetry propagation, q-path, tasks, plotting, the ground-state
    # guards) is unchanged; only the LSWT engine is swapped.
    mode = str(calc_config.get('mode', 'dipole')).upper()
    if mode not in ('DIPOLE', 'SUN'):
        raise ValueError(
            f"calculation.mode must be 'dipole' or 'SUN', got {calc_config.get('mode')!r}.")

    if mode == 'DIPOLE':
        _advise_sun_mode(spin_model)

    logger.info(f"Initializing {'SU(N)' if mode == 'SUN' else 'MagCalc'} "
                f"for LSWT Calculation...")
    try:
        if mode == 'SUN':
            from magcalc.sun.adapter import SUNCalculator
            calculator = SUNCalculator(spin_model, final_config, params_val)
        else:
            calculator = mc.MagCalc(
                spin_model_module=spin_model,
                spin_magnitude=S_val,
                hamiltonian_params=params_val,
                cache_file_base=cache_base,
                cache_mode=cache_mode,
            )
    except Exception as e:
        logger.error(f"Failed to initialize the LSWT engine: {e}")
        # Explicitly re-raise to stop execution and show error in UI
        raise e

    # --- Ground-state stability gate -------------------------------------------
    # LSWT expanded about anything other than a true classical minimum has
    # IMAGINARY magnon energies. Historically this only produced per-q warnings
    # while the run completed and wrote a plausible-looking plot, so a wrong
    # ground state (a local minimum from too few minimization starts, or a
    # hand-written magnetic_structure that is not the minimum) silently yielded
    # wrong physics. Check it ONCE, up front, before spending time on a spectrum
    # that is already invalid.
    #   calculation.on_imaginary: error (default) | warn | off
    #   calculation.imaginary_tolerance: meV (default 1e-4)
    on_imaginary = calc_config.get('on_imaginary', 'error')
    if on_imaginary not in ('error', 'warn', 'off'):
        raise ValueError(
            f"calculation.on_imaginary must be 'error', 'warn' or 'off', "
            f"got {on_imaginary!r}.")
    if on_imaginary != 'off':
        # Guard 2 (independent of the imaginary check): does a downhill step from
        # the current structure find a LOWER energy? This catches the case the
        # imaginary check provably cannot -- a stationary maximum/saddle, e.g. a
        # ferromagnetic structure supplied for an antiferromagnet, whose spectrum
        # comes back real and positive because the +/- omega pairs are sorted and
        # the upper half returned.
        relax = calculator.relax_from_current()
        if relax is not None:
            e_now, e_relaxed = relax
            e_tol = float(calc_config.get('energy_tolerance', 1e-6))
            scale = max(abs(e_now), 1.0)
            if e_relaxed < e_now - max(e_tol, 1e-9 * scale):
                emsg = (
                    f"The magnetic structure is NOT a classical energy minimum: relaxing "
                    f"it downhill lowers the energy from {e_now:.6f} to {e_relaxed:.6f} meV "
                    f"({e_now - e_relaxed:.3e} meV lower).\n"
                    f"LSWT about a non-minimum is meaningless -- and note this case can "
                    f"still return a real, positive-looking spectrum, so it will NOT show "
                    f"up as imaginary energies.\n"
                    f"  * Enable `tasks: {{minimization: true}}` (with enough "
                    f"`num_starts` / `early_stopping`) instead of supplying the structure "
                    f"by hand, or fix the supplied `magnetic_structure`.\n"
                    f"  * If the structure is KNOWINGLY metastable, set "
                    f"`calculation: {{on_imaginary: warn}}` (in Studio: Tasks & Plotting "
                    f"-> Calculation Settings -> Ground-State Check) to downgrade this to "
                    f"a warning, or `off` to disable both guards."
                )
                if on_imaginary == 'error':
                    raise ValueError(emsg)
                logger.warning(emsg)

        imag_tol = float(calc_config.get('imaginary_tolerance', 1e-4))
        # Test the q-path the user is actually going to plot, not just random q: an
        # instability confined to particular q (SW23's acoustic branch collapses only
        # at the magnetic satellites) is easily missed by a random sample -- and those
        # are precisely the q that will end up in the figure.
        q_guard = None
        try:
            q_guard_rlu = q_vectors if q_vectors is not None else \
                generate_q_path_from_config(final_config)
            if q_guard_rlu is not None and len(q_guard_rlu) > 0:
                q_guard = np.dot(np.array(q_guard_rlu), compute_b_matrix(spin_model))
        except Exception:
            logger.debug("Ground-state guard: no q-path available; using random q only.")
        # Threshold on |Im| RELATIVE to the bandwidth. An absolute meV cutoff cannot
        # separate a real instability from numerical noise across models whose energy
        # scales differ by orders of magnitude -- and the noise is worst exactly where
        # it matters, at the omega ~ 0 Goldstone modes where the Bogoliubov problem is
        # singular (SW07's 120-degree kagome: 1e-3 meV of noise on a 2.4 meV band).
        rel_tol = float(calc_config.get('imaginary_rel_tolerance', 5e-3))
        report = calculator.stability_report(q_cart=q_guard)
        max_imag = report["max_imag"]
        if max_imag > imag_tol and report["relative"] > rel_tol:
            msg = (
                f"Magnon energies are IMAGINARY (max |Im(omega)| = {max_imag:.3e} meV, "
                f"{report['relative']:.1%} of the {report['band_scale']:.3g} meV bandwidth).\n"
                f"This almost always means the magnetic structure is NOT the classical "
                f"ground state, so the spin-wave expansion is about the wrong state and "
                f"the spectrum is meaningless.\n"
                f"  * If you are minimizing: raise `minimization.num_starts` and "
                f"`minimization.early_stopping` (the default early_stopping=10 is too "
                f"permissive for large/frustrated cells), then confirm the minimum "
                f"energy is reproducible across several `seed` values.\n"
                f"  * If you supplied `magnetic_structure` by hand: it is not a minimum "
                f"of this Hamiltonian.\n"
                f"  * If the instability is physical and expected, set "
                f"`calculation: {{on_imaginary: warn}}` (in Studio: Tasks & Plotting -> "
                f"Calculation Settings -> Ground-State Check) to downgrade this to a warning."
            )
            if on_imaginary == 'error':
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"Ground-state stability check passed (max |Im(omega)| = "
                f"{max_imag:.2e} meV).")

    # 1b. Data Fitting (optional)
    # Runs before the forward tasks so that, on success, the best-fit parameters
    # are loaded into the SAME calculator (no re-init) and every downstream
    # dispersion/S(Q,w)/powder/plot task renders the best-fit model for direct
    # comparison against the data.
    do_fit = tasks.get('fit', False) or final_config.get('fitting', {}).get('enabled', False)
    if do_fit:
        from magcalc import fitting
        logger.info("Data fitting enabled.")
        try:
            name_order = fitting.canonical_name_order(final_config)
            B_matrix = compute_b_matrix(spin_model)
            fit_out = fitting.run_fit(
                final_config=final_config,
                calculator=calculator,
                name_order=name_order,
                B_matrix=B_matrix,
                config_dir=config_dir,
                backend=backend,
            )

            # Pin the calculator (and params_val) to the optimum for downstream tasks.
            best_p = fit_out['best_p']
            calculator.update_hamiltonian_params(best_p)
            params_val = best_p
            logger.info(f"Best-fit parameters: {fit_out['best_values']}")

            out_cfg = final_config.get('output', {})
            report_file = out_cfg.get('fit_report_filename', 'fit_report.txt')
            if not os.path.isabs(report_file): report_file = os.path.join(config_dir, report_file)
            _safe_makedirs(report_file)
            with open(report_file, 'w') as f:
                f.write(fit_out['report'] + "\n")
            logger.info(f"Fit report written to {report_file}")

            params_file = out_cfg.get('fit_params_filename', 'fit_params.yaml')
            if not os.path.isabs(params_file): params_file = os.path.join(config_dir, params_file)
            _safe_makedirs(params_file)
            with open(params_file, 'w') as f:
                yaml.safe_dump({'best_fit_parameters': fit_out['best_values']}, f,
                               default_flow_style=False)
            logger.info(f"Best-fit parameters written to {params_file}")

            # Optional data-vs-model comparison plot.
            should_plot_fit = tasks.get('plot_fit')
            if should_plot_fit is None:
                should_plot_fit = final_config.get('plotting', {}).get('plot_fit', True)
            if should_plot_fit:
                from magcalc.plotting import plot_fit_comparison
                fit_type = final_config.get('fitting', {}).get('type', 'dispersion')
                prediction = fit_out['problem'].predict(fit_out['result'].params)
                plot_cfg = final_config.get('plotting', {})
                fit_plot = plot_cfg.get('fit_plot_filename', 'fit_comparison.png')
                if not os.path.isabs(fit_plot): fit_plot = os.path.join(config_dir, fit_plot)
                plot_fit_comparison(
                    fit_type=fit_type,
                    prediction=prediction,
                    save_filename=fit_plot,
                    title=plot_cfg.get('fit_title', f"Fit comparison ({fit_type})"),
                    show_plot=plot_cfg.get('show_plot', False),
                )
        except Exception as e:
            logger.error(f"Data fitting failed: {e}")
            raise e

    # Store calculated data for plotting if not saved to file
    memory_cache = {
        'dispersion': None,
        'sqw': None
    }

    save_data_flag = final_config.get('output', {}).get('save_data', True)

    # Emit dispersion outputs (memory cache, .npz, optional CSV) from energies.
    # Shared by the standalone eigensolve and the guarded reuse of S(Q,w) energies.
    def _emit_dispersion(q_cart, energies, disp_file):
        if energies is None:
            return
        try:
            energies_arr = np.array(energies)
        except (ValueError, TypeError):
            energies_arr = np.array(energies, dtype=object)

        memory_cache['dispersion'] = {
            'q_vectors': q_cart,
            'energies': energies_arr
        }

        if save_data_flag:
            _safe_makedirs(disp_file)
            np.savez(disp_file, q_vectors=q_cart, energies=energies_arr)
            logger.info(f"Dispersion saved to {disp_file}")

        if tasks.get('export_csv', False):
            disp_csv = final_config.get('output', {}).get('disp_csv_filename', 'disp_data.csv')
            if not os.path.isabs(disp_csv): disp_csv = os.path.join(config_dir, disp_csv)

            logger.info(f"Exporting dispersion to CSV: {disp_csv}")
            # qx, qy, qz, en1, en2, ...
            header = "qx,qy,qz," + ",".join([f"en{i}" for i in range(energies_arr.shape[1])])
            with open(disp_csv, 'w') as f:
                f.write(header + "\n")
                for i in range(len(q_cart)):
                    q = q_cart[i]
                    en = energies_arr[i]
                    line = f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f}," + ",".join([f"{e:.6f}" for e in en])
                    f.write(line + "\n")

    # Satellite (q +/- k) branches for single-k structures. Config surface:
    # magnetic_structure.satellites, overridable by tasks.satellites.
    # Defaults: S(Q,w) True when a single-k structure is active (that is the
    # physical cross-section), dispersion False (preserves prior output shape).
    single_k_active = getattr(calculator, 'k_cart', None) is not None
    ms_cfg_norm = getattr(spin_model, 'mag_struct_cfg', None) or \
        (final_config.get('magnetic_structure') or {})
    sat_flag = ms_cfg_norm.get('satellites')
    if tasks.get('satellites') is not None:
        sat_flag = tasks.get('satellites')
    disp_satellites = bool(sat_flag) if sat_flag is not None else False
    sqw_satellites = bool(sat_flag) if sat_flag is not None else True
    if single_k_active:
        logger.info(
            f"Single-k structure: satellites for dispersion={disp_satellites}, "
            f"S(Q,w)={sqw_satellites}."
        )

    # 2. Dispersion
    do_dispersion = tasks.get('dispersion', False)
    # When S(Q,w) is also requested it returns identical energies on the same
    # q-path, so defer the dispersion and reuse them rather than running a
    # redundant eigensolve. Set to (q_cart, disp_file) when deferred.
    dispersion_pending = None

    if do_dispersion:
        disp_file = final_config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
        if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)

        need_recalc = tasks.get('calculate_dispersion', True)
        if not os.path.exists(disp_file) and save_data_flag: need_recalc = True

        if need_recalc:
            if q_vectors is None:
                q_vectors = generate_q_path_from_config(final_config)

            if len(q_vectors) > 0:
                B_matrix = compute_b_matrix(spin_model)
                q_vectors_cart = np.dot(np.array(q_vectors), B_matrix)

                # Domain averaging concatenates modes per domain in S(Q,w);
                # dispersion stays single-domain, so its energies cannot be
                # reused from a domain-averaged S(Q,w) result. (temperature /
                # cross_section only rescale intensities — energies unaffected.)
                can_reuse_sqw = ((not single_k_active) or (disp_satellites == sqw_satellites)) \
                    and not domains
                if tasks.get('sqw_map', False) and tasks.get('calculate_sqw_map', True) \
                        and can_reuse_sqw:
                    # S(Q,w) (computed next, same q-path) yields identical
                    # energies; defer and reuse them. Skips a full eigensolve
                    # pass and its worker-pool spin-up. Only valid when both
                    # use the same satellite setting (mode counts match).
                    dispersion_pending = (q_vectors_cart, disp_file)
                    logger.info(
                        "Dispersion deferred to reuse S(Q,w) energies "
                        "(same q-path; skipping redundant eigensolve)."
                    )
                else:
                    logger.info(f"Calculating dispersion... (backend={backend})")
                    disp_res = calculator.calculate_dispersion(
                        q_vectors_cart, backend=backend, satellites=disp_satellites)
                    _emit_dispersion(q_vectors_cart, disp_res.energies, disp_file)
            else:
                logger.warning("No Q-vectors.")
    
    # 3. S(Q,w) Map
    do_sqw = tasks.get('sqw_map', False)
    
    if do_sqw:
        sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
        if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
        
        need_recalc = tasks.get('calculate_sqw_map', True)
        if not os.path.exists(sqw_file) and save_data_flag: need_recalc = True
        
        if need_recalc:
            if q_vectors is None:
                q_vectors = generate_q_path_from_config(final_config)
            
            if q_vectors is not None and len(q_vectors) > 0:
                B_matrix = compute_b_matrix(spin_model)
                q_vectors_cart = np.dot(q_vectors, B_matrix)

                logger.info(f"Calculating S(Q,w)... (backend={backend})")
                sqw_res = calculator.calculate_sqw(
                    q_vectors_cart, backend=backend, satellites=sqw_satellites,
                    temperature=temperature, domains=domains,
                    cross_section=cross_section)
                q_out = sqw_res.q_vectors
                en_out = sqw_res.energies
                int_out = sqw_res.intensities
                
                memory_cache['sqw'] = {
                    'q_vectors': q_out,
                    'energies': en_out,
                    'intensities': int_out
                }

                if save_data_flag:
                    _safe_makedirs(sqw_file)
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

    # Guarded reuse: derive the deferred dispersion from the S(Q,w) energies
    # just computed (identical on the shared q-path). Fall back to a standalone
    # eigensolve if S(Q,w) produced no result, so the output is always emitted.
    if dispersion_pending is not None:
        q_cart_disp, disp_file_disp = dispersion_pending
        if memory_cache.get('sqw') is not None:
            _emit_dispersion(
                memory_cache['sqw']['q_vectors'],
                memory_cache['sqw']['energies'],
                disp_file_disp,
            )
            logger.info("Dispersion reused from S(Q,w) result (no extra eigensolve).")
        else:
            logger.info(f"S(Q,w) result unavailable; computing dispersion directly. (backend={backend})")
            disp_res = calculator.calculate_dispersion(
                q_cart_disp, backend=backend, satellites=disp_satellites)
            _emit_dispersion(q_cart_disp, disp_res.energies, disp_file_disp)

    # 3c. 1/S (LSWT) corrections -- zero-point energy and ordered-moment reduction.
    if tasks.get('corrections', False):
        from magcalc.corrections import compute_corrections
        cc = final_config.get('corrections', {}) or {}
        mesh = cc.get('k_mesh', [16, 16, 16])
        try:
            res = compute_corrections(calculator, k_mesh=tuple(mesh))
            logger.info(
                f"1/S corrections ({res.n_kpoints} k-points): "
                f"energy {res.energy_correction_per_site:+.6f} meV/site; "
                f"ordered-moment reduction dS = "
                f"{np.array2string(res.moment_reduction, precision=4)} "
                f"(<S^z> = S - dS).")
            memory_cache['corrections'] = {
                'energy_correction_per_site': res.energy_correction_per_site,
                'moment_reduction': res.moment_reduction,
                'n_kpoints': res.n_kpoints,
            }
            if save_data_flag:
                cf = final_config.get('output', {}).get(
                    'corrections_filename', 'corrections.npz')
                if not os.path.isabs(cf):
                    cf = os.path.join(config_dir, cf)
                _safe_makedirs(cf)
                np.savez(cf, energy_correction_per_site=res.energy_correction_per_site,
                         moment_reduction=res.moment_reduction, n_kpoints=res.n_kpoints)
                logger.info(f"1/S corrections saved to {cf}")
        except Exception as e:
            logger.error(f"1/S correction calculation failed: {e}")
            raise

    # 4. Powder Average
    # NOTE on units: dispersion and S(Q,w) above interpret q_path entries as
    # reciprocal-lattice units (RLU) and multiply by the B-matrix internally,
    # but the powder section consumes |Q| magnitudes in absolute reciprocal
    # angstrom (1/A). They are NOT converted via B-matrix. Configure
    # powder_average.q_min/q_max/q_count or powder_average.q_magnitudes
    # accordingly.
    do_powder = tasks.get('powder_average', False)
    if do_powder:
        powder_file = final_config.get('output', {}).get('powder_data_filename', 'powder_data.npz')
        if not os.path.isabs(powder_file): powder_file = os.path.join(config_dir, powder_file)

        powder_config = final_config.get('powder_average', {})
        q_mags = powder_config.get('q_magnitudes')
        if q_mags is None:
            q_min = powder_config.get('q_min', 0.1)
            q_max = powder_config.get('q_max', 4.0)
            q_count = powder_config.get('q_count', 50)
            q_mags = np.linspace(q_min, q_max, q_count)
            
        num_samples = powder_config.get('num_samples', 50)
        
        logger.info(f"Calculating Powder Average (Q={q_mags[0]:.2f} to {q_mags[-1]:.2f}, {len(q_mags)} points, {num_samples} samples, backend={backend})...")
        powder_res = calculator.calculate_powder_average(
            q_mags, num_samples=num_samples, backend=backend,
            temperature=temperature, cross_section=cross_section)
        
        if powder_res:
            memory_cache['powder'] = {
                'q_vectors': powder_res.q_vectors,
                'energies': powder_res.energies,
                'intensities': powder_res.intensities
            }
            if save_data_flag:
                _safe_makedirs(powder_file)
                np.savez(powder_file, q_vectors=powder_res.q_vectors, energies=powder_res.energies, intensities=powder_res.intensities)
                logger.info(f"Powder average data saved to {powder_file}")

    # 4b. Constant-energy cuts on a 2-D q grid (SW10-style; first-class
    # replacement for the old companion scripts). Config:
    #   tasks: {energy_cut: true}
    #   energy_cut:
    #     origin: [0, 0, 0]                      # RLU corner of the grid
    #     axis1: {vec: [2, 0, 0], points: 121}   # RLU span from origin
    #     axis2: {vec: [0, 2, 0], points: 121}
    #     cuts:
    #       - {center: 3.75, fwhm: 0.25}         # Gaussian energy window
    #       - {band: [3.5, 4.01]}                # hard integration window
    # Intensities inherit temperature / domains / cross_section from
    # `calculation`, and satellites from the magnetic structure.
    do_ecut = tasks.get('energy_cut', False)
    if do_ecut:
        ec = final_config.get('energy_cut', {}) or {}
        cuts_spec = ec.get('cuts', [])
        if not cuts_spec:
            logger.warning("tasks.energy_cut is on but energy_cut.cuts is empty; skipping.")
        else:
            origin = np.asarray(ec.get('origin', [0.0, 0.0, 0.0]), dtype=float)
            ax1 = ec.get('axis1', {}) or {}
            ax2 = ec.get('axis2', {}) or {}
            v1 = np.asarray(ax1.get('vec', [1.0, 0.0, 0.0]), dtype=float)
            v2 = np.asarray(ax2.get('vec', [0.0, 1.0, 0.0]), dtype=float)
            n1 = int(ax1.get('points', 51))
            n2 = int(ax2.get('points', 51))
            s1 = np.linspace(0.0, 1.0, n1)
            s2 = np.linspace(0.0, 1.0, n2)
            S1, S2 = np.meshgrid(s1, s2, indexing='ij')
            q_rlu = (origin[None, :]
                     + S1.ravel()[:, None] * v1[None, :]
                     + S2.ravel()[:, None] * v2[None, :])
            B_matrix = compute_b_matrix(spin_model)
            q_grid_cart = q_rlu @ B_matrix

            logger.info(
                f"Calculating energy cut(s) on a {n1}x{n2} q grid "
                f"({q_grid_cart.shape[0]} points, backend={backend})...")
            ecut_res = calculator.calculate_sqw(
                q_grid_cart, backend=backend, satellites=sqw_satellites,
                temperature=temperature, domains=domains,
                cross_section=cross_section)
            if ecut_res is None:
                logger.error("Energy-cut S(Q,w) calculation failed.")
            else:
                E_g = np.asarray(ecut_res.energies)
                I_g = np.asarray(ecut_res.intensities)
                panels, labels = [], []
                for cut in cuts_spec:
                    if 'center' in cut:
                        center = float(cut['center'])
                        fwhm = float(cut.get('fwhm', 0.25))
                        sigma = fwhm / 2.3548200450309493
                        with np.errstate(invalid='ignore'):
                            Z = np.nansum(
                                I_g * np.exp(-((E_g - center) ** 2) / (2 * sigma ** 2)),
                                axis=1)
                        labels.append(f"E = {center} meV (FWHM {fwhm} meV)")
                    elif 'band' in cut:
                        lo, hi = (float(cut['band'][0]), float(cut['band'][1]))
                        with np.errstate(invalid='ignore'):
                            Z = np.nansum(I_g * ((E_g >= lo) & (E_g < hi)), axis=1)
                        labels.append(f"integrated {lo}-{hi} meV")
                    else:
                        logger.warning(f"Skipping energy cut {cut}: needs 'center' or 'band'.")
                        continue
                    panels.append(Z.reshape(n1, n2))

                coords1, label1 = _grid_axis_coords(origin, v1, s1)
                coords2, label2 = _grid_axis_coords(origin, v2, s2)

                ecut_file = final_config.get('output', {}).get(
                    'energy_cut_data_filename', 'energy_cut_data.npz')
                if not os.path.isabs(ecut_file):
                    ecut_file = os.path.join(config_dir, ecut_file)
                if save_data_flag:
                    _safe_makedirs(ecut_file)
                    np.savez(ecut_file,
                             q_rlu=q_rlu, coords1=coords1, coords2=coords2,
                             panels=np.array(panels), labels=np.array(labels))
                    logger.info(f"Energy-cut data saved to {ecut_file}")

                plot_cfg_early = final_config.get('plotting', {}) or {}
                if plot_cfg_early.get('save_plot', True) and panels:
                    ecut_plot = plot_cfg_early.get(
                        'energy_cut_plot_filename', 'energy_cut.png')
                    if not os.path.isabs(ecut_plot):
                        ecut_plot = os.path.join(config_dir, ecut_plot)
                    plot_energy_cuts(
                        coords1, coords2, panels, labels,
                        save_filename=ecut_plot,
                        axis_labels=(label1, label2),
                        title=plot_cfg_early.get('energy_cut_title', "Constant-energy cuts"),
                        cmap=plot_cfg_early.get('cmap', 'viridis'),
                        show_plot=plot_cfg_early.get('show_plot', False),
                    )

    # 5. Plotting
    plot_config = final_config.get('plotting', {})
    
    # Priority: tasks['plot_dispersion'] -> tasks['run_plotting'] -> (do_dispersion and save_plot)
    should_plot_disp = tasks.get('plot_dispersion')
    if should_plot_disp is None:
        should_plot_disp = tasks.get('run_plotting')
    if should_plot_disp is None:
        should_plot_disp = do_dispersion and plot_config.get('save_plot', True)
    
    # Priority: tasks['plot_sqw_map'] -> tasks['run_plotting'] -> (do_sqw and save_plot)
    should_plot_sqw = tasks.get('plot_sqw_map')
    if should_plot_sqw is None:
        should_plot_sqw = tasks.get('run_plotting')
    if should_plot_sqw is None:
        should_plot_sqw = do_sqw and plot_config.get('save_plot', True)

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
                 show_plot=plot_config.get('show_plot', False),
                 auto_scale=plot_config.get('auto_scale_disp', True)
             )
        else:
            disp_file = final_config.get('output', {}).get('disp_data_filename', 'disp_data.npz')
            if not os.path.isabs(disp_file): disp_file = os.path.join(config_dir, disp_file)
            
            if os.path.exists(disp_file):
                data = np.load(disp_file)
                plot_filename = plot_config.get('disp_plot_filename', 'disp_plot.png')
                if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                
                plot_dispersion(
                    q_vectors=data['q_vectors'],
                    energies=data['energies'],
                    save_filename=plot_filename,
                    title=plot_config.get('disp_title', "Dispersion"),
                    ylim=plot_config.get('energy_limits_disp'),
                    show_plot=plot_config.get('show_plot', False),
                    auto_scale=plot_config.get('auto_scale_disp', True)
                )

    if should_plot_sqw:
        # Prefer memory cache
        if memory_cache['sqw'] is not None and do_sqw:
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
                show_plot=plot_config.get('show_plot', False),
                resolution=plot_config.get('resolution'),
                energy_grid_step=plot_config.get('energy_grid_step', 0.05)
            )
        else:
            sqw_file = final_config.get('output', {}).get('sqw_data_filename', 'sqw_data.npz')
            if not os.path.isabs(sqw_file): sqw_file = os.path.join(config_dir, sqw_file)
            
            if os.path.exists(sqw_file):
                data = np.load(sqw_file)
                plot_filename = plot_config.get('sqw_plot_filename', 'sqw_plot.png')
                if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                
                plot_ints = data.get('intensities')
                if plot_ints is None: plot_ints = data.get('sqw_values')

                plot_sqw_map(
                    q_vectors=data['q_vectors'],
                    energies=data['energies'],
                    intensities=plot_ints,
                    save_filename=plot_filename,
                    title=plot_config.get('sqw_title', "S(Q,w)"),
                    ylim=plot_config.get('energy_limits_sqw'),
                    broadening_width=plot_config.get('broadening_width', 0.2),
                    cmap=plot_config.get('cmap', 'PuBu_r'),
                    show_plot=plot_config.get('show_plot', False),
                    resolution=plot_config.get('resolution'),
                    energy_grid_step=plot_config.get('energy_grid_step', 0.05)
                )

    # 6. Powder Plotting
    should_plot_powder = tasks.get('run_powder_average')
    if should_plot_powder is None:
        should_plot_powder = do_powder and plot_config.get('save_plot', True)

    if should_plot_powder:
        # Prefer memory cache
        if memory_cache.get('powder') is not None:
            data = memory_cache['powder']
            plot_filename = plot_config.get('powder_plot_filename', 'powder_plot.png')
            if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
            
            plot_sqw_map(
                q_vectors=data['q_vectors'],
                energies=data['energies'],
                intensities=data['intensities'],
                save_filename=plot_filename,
                title=plot_config.get('powder_title', "Powder Average S(Q,w)"),
                ylim=plot_config.get('energy_limits_sqw'),
                broadening_width=plot_config.get('broadening_width', 0.2),
                cmap=plot_config.get('cmap', 'PuBu_r'),
                show_plot=plot_config.get('show_plot', False),
                resolution=plot_config.get('resolution'),
                x_is_qmag=True,
                energy_grid_step=plot_config.get('energy_grid_step', 0.05)
            )
        else:
            powder_file = final_config.get('output', {}).get('powder_data_filename', 'powder_data.npz')
            if not os.path.isabs(powder_file): powder_file = os.path.join(config_dir, powder_file)
            
            if os.path.exists(powder_file):
                data = np.load(powder_file)
                plot_filename = plot_config.get('powder_plot_filename', 'powder_plot.png')
                if not os.path.isabs(plot_filename): plot_filename = os.path.join(config_dir, plot_filename)
                
                plot_sqw_map(
                    q_vectors=data['q_vectors'],
                    energies=data['energies'],
                    intensities=data['intensities'],
                    save_filename=plot_filename,
                    title=plot_config.get('powder_title', "Powder Average S(Q,w)"),
                    ylim=plot_config.get('energy_limits_sqw'),
                    broadening_width=plot_config.get('broadening_width', 0.2),
                    cmap=plot_config.get('cmap', 'PuBu_r'),
                    show_plot=plot_config.get('show_plot', False),
                    resolution=plot_config.get('resolution'),
                    x_is_qmag=True,
                    energy_grid_step=plot_config.get('energy_grid_step', 0.05)
                )
