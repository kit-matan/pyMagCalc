// The editor's blank-slate state. Lives here (not in App.jsx) because
// importConfigDoc merges a loaded file over it, and the round-trip test needs
// the same starting point the app uses.
export const DEFAULT_CONFIG = {
  lattice: { a: 5.0, b: 5.0, c: 5.0, alpha: 90, beta: 90, gamma: 90, space_group: 1 },
  wyckoff_atoms: [],
  magnetic_elements: ["Cu"],
  symmetry_interactions: [],
  explicit_interactions: [],
  parameters: { H_mag: 0.0, H_dir: [0, 0, 1] },
  tasks: {
    minimization: true,
    dispersion: true,
    plot_dispersion: true,
    sqw_map: false,
    plot_sqw_map: false,
    powder_average: false,
    export_csv: false,
    plot_structure: false,
    corrections: false,
    scga: false,
    thermal_mc: false,
    sampled_correlations: false,
    kpm_sqw: false
  },
  q_path: {
    points: { Gamma: [0, 0, 0] },
    path: ['Gamma'],
    points_per_segment: 100
  },
  plotting: {
    energy_min: 0,
    energy_max: 20,
    broadening: 0.2,
    energy_resolution: 0.05,
    momentum_max: 4.0,
    auto_scale_disp: true,
    save_plot: true,
    disp_plot_filename: 'disp_plot.png',
    sqw_plot_filename: 'sqw_plot.png',
    show_plot: true,
    plot_structure: false
  },
  output: {
    disp_data_filename: 'disp_data.npz',
    sqw_data_filename: 'sqw_data.npz',
    disp_csv_filename: 'disp_data.csv',
    sqw_csv_filename: 'sqw_data.csv',
    save_data: true
  },
  magnetic_structure: {
    enabled: false,
    type: 'pattern',
    pattern_type: 'antiferromagnetic',
    directions: []
  },
  minimization: {
    // Monte-Carlo annealing (SpinW `anneal` / Sunny LocalSampler) is the default:
    // it crosses barriers, so it does not get trapped the way multistart gradient
    // descent does. On SW20-in-field the gradient path reached the true minimum in
    // only 3 of 200 starts; anneal finds it in a single run.
    method: "anneal",
    num_starts: 4,
    n_sweeps: 2000,
    n_workers: 8,
    early_stopping: 10
  },
  powder_average: {
    q_min: 0.1,
    q_max: 4.0,
    q_count: 50,
    num_samples: 50
  },
  calculation: {
    // 'auto' reuses the cached symbolic matrix (gen_HM) across runs; it is
    // deterministic per model topology, so regenerating it every run ('none')
    // needlessly costs seconds of cold-start time. Measured ~79x faster on a
    // 9-spin model. Cache auto-invalidates when the model structure changes.
    cache_mode: 'auto',
    backend: 'numpy',
    // LSWT is an expansion about a classical energy MINIMUM. If the magnetic
    // structure is not one, the spectrum is meaningless -- so the run FAILS by
    // default rather than drawing a plausible-looking plot. 'warn' is for
    // structures that are knowingly metastable (e.g. a commensurate approximation
    // to an incommensurate spiral).
    on_imaginary: 'error',
    mode: 'dipole',
    temperature: null,
    cross_section: 'perp',
    // entangled mode extras (used only when mode === 'entangled')
    series_order: 0,
    units_text: ''
  },
  // Beyond-LSWT task settings (sent only when the matching task is enabled).
  scga: { temperature: 1.0, mesh_density: 12 },
  thermal_mc: { temperatures: '0.5, 1.0, 2.0, 4.0', supercell: '4, 4, 1',
                n_sweeps: 4000, n_equil: 1500 },
  sampled_correlations: { temperature: 0.5, supercell: '8, 1, 1', dt: 0.02,
                          n_steps: 2048, n_traj: 8 },
  kpm: { e_min: 0.0, e_max: 10.0, e_step: 0.05, fwhm: 0.1 },
  fitting: {
    type: 'dispersion',
    data_file: '',
    data_label: '',
    method: 'leastsq',
    vary: [],
    bounds: {},
    match: 'nearest',
    scale: { value: 1.0, vary: true },
    background: { value: 0.0, vary: true },
    energy_broadening: { value: 0.3, vary: false }
  }
}
