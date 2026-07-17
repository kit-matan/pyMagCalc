import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, EyeOff, Share2, Info, Magnet, Wind, Check, ChevronRight, Zap, Crosshair, FileText, BarChart2, Play, Image, ArrowDown, X, XCircle, Minus, ChevronDown, Search, Square, Target } from 'lucide-react'
import spaceGroupsList from './data/space_groups.json';
import yaml from 'js-yaml'
import Visualizer from './components/Visualizer'
import MagneticStructureViewer from './components/MagneticStructureViewer'
import LogConsole from './components/LogConsole'
import AppHeader from './components/AppHeader'
import Sidebar from './components/Sidebar'
import { calculateExchangeMatrixSymbolic } from './lib/exchangeMatrix'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('structure')
  const [jsonCache, setJsonCache] = useState({})

  const loadStructureData = async (url) => {
    try {
      const res = await fetch(`${url}?t=${Date.now()}`);
      const data = await res.json();
      setJsonCache(prev => ({ ...prev, [url]: data }));
    } catch (e) {
      console.error("Failed to load structure JSON", e);
    }
  };
  const [showVisualizer, setShowVisualizer] = useState(true)
  const [notification, setNotification] = useState(null)
  const [neighborDistances, setNeighborDistances] = useState([])
  const [selectedBondIdxs, setSelectedBondIdxs] = useState({}) // { suggestionIdx: bondIdx }
  const [selectedBond, setSelectedBond] = useState(null) // Bond clicked in Visualizer
  const [interactionMenuOpen, setInteractionMenuOpen] = useState(false) // Dropdown menu state
  const [interactionMode, setInteractionMode] = useState('symmetry') // 'symmetry' or 'explicit'
  const [atomMode, setAtomMode] = useState('symmetry') // 'symmetry' or 'explicit'
  const [previewAtoms, setPreviewAtoms] = useState([]) // Expanded atoms for visualizer
  const [bonds, setBonds] = useState([]) // Bonds for visualizer
  const [hiddenBondLabels, setHiddenBondLabels] = useState(new Set()) // Labels of bonds to hide
  const [isAddingParam, setIsAddingParam] = useState(false)
  const [newParamName, setNewParamName] = useState('')
  const [calcLoading, setCalcLoading] = useState(false)
  const [calcResults, setCalcResults] = useState(null)

  /* Space Group Search State */
  const [sgSearch, setSgSearch] = useState("");
  const [isSgDropdownOpen, setIsSgDropdownOpen] = useState(false);
  const sgDropdownRef = React.useRef(null);

  const [calcError, setCalcError] = useState(null)
  const [calcStopping, setCalcStopping] = useState(false)
  // Tracks a user-initiated stop so the resulting aborted request isn't shown as an error.
  const stopRequestedRef = React.useRef(false)
  // Holds the crystal_structure / interactions / magnetic_structure of a loaded
  // example config verbatim, so it is sent to the backend exactly as the CLI
  // reads it (lattice_vectors, interaction_matrix, single_ion_anisotropy,
  // spiral/generic magnetic structures, ...) instead of being flattened into
  // the designer's a/b/c + Heisenberg model. Cleared on reset / CIF load.
  const rawImportRef = React.useRef(null)

  // Resizable layout state
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [visualizerWidth, setVisualizerWidth] = useState(450)
  const resizingRef = React.useRef(null) // 'left' or 'right'

  const startResizing = (direction) => (e) => {
    e.preventDefault()
    resizingRef.current = direction
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const stopResizing = () => {
    if (resizingRef.current) {
      resizingRef.current = null
      document.body.style.cursor = 'default'
      document.body.style.userSelect = 'auto'
    }
  }

  const resize = React.useCallback((e) => {
    if (!resizingRef.current) return

    if (resizingRef.current === 'left') {
      // Limit sidebar width: min 200, max 600
      let newWidth = e.clientX
      if (newWidth < 200) newWidth = 200
      if (newWidth > 600) newWidth = 600
      setSidebarWidth(newWidth)
    } else if (resizingRef.current === 'right') {
      // Calculate from right edge
      let newWidth = window.innerWidth - e.clientX
      if (newWidth < 300) newWidth = 300
      if (newWidth > 800) newWidth = 800
      setVisualizerWidth(newWidth)
    }
  }, [])

  React.useEffect(() => {
    window.addEventListener('mousemove', resize)
    window.addEventListener('mouseup', stopResizing)
    return () => {
      window.removeEventListener('mousemove', resize)
      window.removeEventListener('mouseup', stopResizing)
    }
  }, [resize])
  const [logs, setLogs] = useState([])
  const [wsConnected, setWsConnected] = useState(false)

  // WebSocket Log Connection
  React.useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;
    let isMounted = true;

    const connect = () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
        console.log("Connecting log WS:", wsUrl);

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log("Log WebSocket Connected");
          setWsConnected(true);
          
          // Heartbeat to keep connection alive
          const heartbeat = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, 30000);
          
          ws._heartbeat = heartbeat;
        };

        ws.onmessage = (event) => {
          const rawMsg = event.data;
          if (rawMsg === 'pong' || rawMsg === 'ping') return;
          
          setLogs(prev => {
            if (rawMsg.startsWith('\r') && prev.length > 0) {
              const next = [...prev];
              next[next.length - 1] = rawMsg.replace(/^\r/, '');
              return next;
            }
            const newLogs = [...prev, rawMsg];
            if (newLogs.length > 1000) return newLogs.slice(newLogs.length - 1000);
            return newLogs;
          });
        };

        ws.onclose = () => {
          console.log("Log WebSocket Closed. Retrying...");
          setWsConnected(false);
          if (ws._heartbeat) clearInterval(ws._heartbeat);
          
          if (isMounted) {
            reconnectTimeout = setTimeout(connect, 5000);
          }
        };

        ws.onerror = (err) => {
          console.error("Log WebSocket Error:", err);
          ws.close();
        };

      } catch (e) {
        console.error("Failed to init WebSocket:", e);
      }
    };

    connect();

    return () => {
      isMounted = false;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) {
        ws.onclose = null; // Prevent reconnect logic on cleanup
        ws.close();
      }
    }
  }, []);

  // Helper to consistently generate keys for bond values (handling arrays and strings)
  const getBondKey = (val) => {
    if (Array.isArray(val)) {
      return val.join(',');
    }
    return String(val);
  }

  const toggleBondVisibility = (label) => {
    // If label is an object/array, stringify? Usually simple string for Heisenberg.
    // Be careful with objects.
    const key = getBondKey(label);
    const next = new Set(hiddenBondLabels);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setHiddenBondLabels(next);
  }


  // Theme Detection
  const [isDark, setIsDark] = useState(window.matchMedia('(prefers-color-scheme: dark)').matches)

  React.useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const listener = (e) => setIsDark(e.matches)
    media.addEventListener('change', listener)
    return () => media.removeEventListener('change', listener)
  }, [])


  const showNotify = (msg, type = 'success') => {
    console.log(`[Notification] ${type}: ${msg}`)
    setNotification({ msg, type })
    setTimeout(() => setNotification(null), 5000)
  }


  const DEMO_CONFIG = {
    lattice: { a: 20.645, b: 8.383, c: 6.442, alpha: 90, beta: 90, gamma: 90, space_group: 43 },
    wyckoff_atoms: [
      { label: 'Cu', pos: [0.16572, 0.3646, 0.7545], spin_S: 0.5 }
    ],
    magnetic_elements: ["Cu"],
    symmetry_interactions: [
      { type: 'heisenberg', ref_pair: ['Cu0', 'Cu2'], distance: 3.1325, value: 'J1', offset: [0, 0, 0] },
      { type: 'dm', ref_pair: ['Cu0', 'Cu2'], distance: 3.1325, value: ['Dx', '0', '0'], offset: [0, 0, 0] },
      { type: 'anisotropic_exchange', ref_pair: ['Cu0', 'Cu2'], distance: 3.1325, value: ['G1', '-G1', '-G1'], offset: [0, 0, 0] },
      { type: 'heisenberg', ref_pair: ['Cu0', 'Cu13'], distance: 3.9751, value: 'J2', offset: [0, 0, 0] },
      { type: 'heisenberg', ref_pair: ['Cu0', 'Cu9'], distance: 5.2572, value: 'J3', offset: [0, 0, 0] }
    ],
    explicit_interactions: [],
    single_ion_anisotropy: [],
    parameters: { J1: 2.49, J2: 2.79, J3: 5.05, G1: 0.28, Dx: 2.67, D: 0.0, H_mag: 20.0, H_dir: [0, 0, 1] },
    tasks: {
      minimization: true,
      dispersion: true,
      plot_dispersion: true,
      sqw_map: true,
      plot_sqw_map: true,
      export_csv: false,
      powder_average: false,
      plot_structure: false,
      corrections: false,
      scga: false,
      thermal_mc: false,
      sampled_correlations: false,
      kpm_sqw: false
    },
    q_path: {
      points: { Start: [0, 1, 0], End: [0, 3, 0] },
      path: ['Start', 'End'],
      points_per_segment: 200
    },
    plotting: {
      energy_min: 0,
      energy_max: 10,
      broadening: 0.2,
      energy_resolution: 0.05,
      momentum_max: 4.0,
      auto_scale_disp: true,
      save_plot: false,
      show_plot: false,
      plot_structure: false
    },
    output: {
      disp_csv_filename: 'disp_data.csv',
      sqw_csv_filename: 'sqw_data.csv'
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
    powder_average: {
      q_min: 0.1,
      q_max: 4.0,
      q_count: 50,
      num_samples: 50
    },
    fitting: {
      type: 'dispersion',          // dispersion | sqw | powder
      data_file: '',               // set by the data-file upload
      data_label: '',              // original filename, for display
      method: 'leastsq',
      vary: [],                    // names of parameters to optimize
      bounds: {},                  // { paramName: [min, max] }
      match: 'nearest',            // dispersion: nearest | mode
      scale: { value: 1.0, vary: true },          // sqw/powder
      background: { value: 0.0, vary: true },      // sqw/powder
      energy_broadening: { value: 0.3, vary: false } // sqw/powder
    }
  }

  const [config, setConfig] = useState(() => {
    try {
      const saved = localStorage.getItem('magcalc_config');
      if (saved) {
        const parsed = JSON.parse(saved);
        // One-time migration: the old default cache_mode was 'none', which made
        // every run regenerate the symbolic matrix (slow cold start each time).
        // 'auto' is now the default and is correctness-safe (the symbolic cache
        // is keyed on the ground-state rotations), so upgrade saved configs that
        // still carry the stale 'none'. Guarded by a flag so a *deliberate*
        // later choice of 'none' from the UI isn't repeatedly overridden.
        if (!localStorage.getItem('magcalc_cache_migrated')) {
          if (parsed.calculation && parsed.calculation.cache_mode === 'none') {
            parsed.calculation.cache_mode = 'auto';
          }
          localStorage.setItem('magcalc_cache_migrated', '1');
        }
        // Merge over defaults so top-level keys added in newer versions (e.g.
        // `fitting`) are always present, even for configs saved before they
        // existed. Without this, accessing config.fitting.* throws and blanks
        // the panel.
        return { ...DEMO_CONFIG, ...parsed };
      }
    } catch (e) {
      console.error("Failed to load config from localStorage", e);
    }
    return DEMO_CONFIG;
  })

  // Build the crystal_structure / interactions / magnetic_structure portion of
  // any backend payload. When an example config was loaded via "Load YAML", its
  // structure and interactions are passed through verbatim (so lattice_vectors,
  // interaction_matrix, single_ion_anisotropy, DM, spiral/generic magnetic
  // structures all reach the backend exactly as `python -m magcalc run` sees
  // them). Otherwise the designer state is serialised as before.
  const buildStructPayload = () => {
    const raw = rawImportRef.current
    if (raw && raw.crystal_structure) {
      const cs = raw.crystal_structure
      const atoms = cs.atoms_uc || cs.wyckoff_atoms || []
      return {
        crystal_structure: {
          ...(cs.lattice_vectors
            ? { lattice_vectors: cs.lattice_vectors }
            : { lattice_parameters: cs.lattice_parameters || config.lattice }),
          atoms_uc: atoms,
          wyckoff_atoms: atoms,
          // A raw config that lists `atoms_uc` is already the full cell (explicit);
          // only `wyckoff_atoms` needs symmetry expansion. Keying off lattice_vectors
          // alone mislabelled lattice_parameters+atoms_uc configs (no space group) as
          // "symmetry", which then tried to re-expand them and dropped per-atom data
          // like the dipole g-tensor.
          atom_mode: cs.atom_mode
            || (cs.atoms_uc ? 'explicit' : (cs.wyckoff_atoms ? 'symmetry'
                : (cs.lattice_vectors ? 'explicit' : 'symmetry'))),
          magnetic_elements: cs.magnetic_elements || config.magnetic_elements || ['Cu'],
          // A magnetic supercell (SpinW nExt / Sunny resize_supercell; also the
          // non-diagonal SU(N) matrix) is physics the runner needs -- pass it through
          // so supercell/SU(N) configs run the same as `magcalc run`.
          ...(cs.magnetic_supercell ? { magnetic_supercell: cs.magnetic_supercell } : {}),
          dimensionality: 3,
        },
        interactions: raw.interactions,
        magnetic_structure: raw.magnetic_structure ?? config.magnetic_structure,
        parameter_order: raw.parameter_order,
      }
    }
    return {
      crystal_structure: {
        ...(config.lattice.lattice_vectors
          ? { lattice_vectors: config.lattice.lattice_vectors }
          : { lattice_parameters: config.lattice }),
        wyckoff_atoms: config.wyckoff_atoms,
        atom_mode: atomMode,
        magnetic_elements: config.magnetic_elements || ['Cu'],
        dimensionality: 3,
      },
      interactions: interactionMode === 'explicit'
        ? { list: config.explicit_interactions || [] }
        : { symmetry_rules: config.symmetry_interactions, single_ion_anisotropy: config.single_ion_anisotropy || [] },
      magnetic_structure: config.magnetic_structure,
      parameter_order: undefined,
    }
  }

  // Persistence Effect
  React.useEffect(() => {
    localStorage.setItem('magcalc_config', JSON.stringify(config));
  }, [config]);

  // Symmetry Expansion Effect for Visualizer
  React.useEffect(() => {
    const updatePreview = async () => {
      try {
        const sp = buildStructPayload()
        const payload = {
          data: {
            crystal_structure: sp.crystal_structure,
            interactions: sp.interactions,
            magnetic_structure: sp.magnetic_structure,
            parameters: config.parameters
          }
        }

        const response = await fetch('/api/get-visualizer-data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })

        if (response.ok) {
          const data = await response.json()
          setPreviewAtoms(data.atoms || [])
          setBonds(data.bonds || [])
        } else {
          // Fallback
          console.error("Visualizer fetch failed")
        }
      } catch (err) {
        console.error('Error expanding structure for preview:', err)
      }
    }
    // Debounce slightly
    const timer = setTimeout(updatePreview, 500)
    return () => clearTimeout(timer)
  }, [config.lattice, config.wyckoff_atoms, atomMode, config.symmetry_interactions, config.explicit_interactions, config.parameters, interactionMode])

  const exportPython = () => {
    let script = `from magcalc.config_builder import MagCalcConfigBuilder\n\n`
    script += `builder = MagCalcConfigBuilder()\n`
    script += `builder.set_lattice(a=${config.lattice.a}, b=${config.lattice.b}, c=${config.lattice.c}, alpha=${config.lattice.alpha}, beta=${config.lattice.beta}, gamma=${config.lattice.gamma}, space_group=${config.lattice.space_group})\n\n`

    config.wyckoff_atoms.forEach(atom => {
      script += `builder.add_wyckoff_atom(label="${atom.label}", pos=[${atom.pos}], spin=${atom.spin_S})\n`
    })

    config.symmetry_interactions.forEach(inter => {
      if (inter.type === 'heisenberg') {
        script += `builder.add_interaction_rule(type="heisenberg", distance=${inter.distance}, value="${inter.value}")\n`
      }
    })

    script += `\nbuilder.save("generated_config.yaml")\n`

    const blob = new Blob([script], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'build_config.py'
    link.click()
  }

  const handleCifUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/parse-cif', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Failed to parse CIF')

      const data = await response.json()
      rawImportRef.current = null  // CIF starts a fresh designer-built structure
      setConfig(prev => ({
        ...prev,
        lattice: data.lattice,
        wyckoff_atoms: data.wyckoff_atoms
      }))
      alert(`CIF Loaded: ${data.international} (SG ${data.lattice.space_group})`)
    } catch (err) {
      alert('Error loading CIF: ' + err.message)
    }
  }

  // Load a magnetic CIF (mCIF): the backend expands the magnetic space group into
  // the full magnetic cell, so we get EXPLICIT atoms (P1) plus a per-site `generic`
  // magnetic structure. The user still supplies the interactions.
  const handleMcifUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/parse-mcif', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        let detail = 'Failed to parse mCIF'
        try { detail = (await response.json()).detail || detail } catch { /* keep default */ }
        throw new Error(detail)
      }

      const data = await response.json()
      rawImportRef.current = null       // mCIF starts a fresh explicit structure
      setAtomMode('explicit')           // magnetic cell is already fully expanded
      setConfig(prev => ({
        ...prev,
        lattice: data.lattice,
        wyckoff_atoms: data.wyckoff_atoms,
        magnetic_elements: data.magnetic_elements ?? prev.magnetic_elements,
        magnetic_structure: { ...prev.magnetic_structure, ...data.magnetic_structure },
      }))
      alert(`mCIF Loaded: ${data.n_sites} magnetic site${data.n_sites === 1 ? '' : 's'} `
            + `(${data.international}). Spin directions imported; now add interactions.`)
    } catch (err) {
      alert('Error loading mCIF: ' + err.message)
    }
  }

  const DEFAULT_CONFIG = {
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

  const resetToDefaults = () => {
    if (window.confirm("Are you sure you want to load the default example (aCVO)?\nCurrent changes will be lost.")) {
      rawImportRef.current = null;
      setConfig(DEMO_CONFIG);
      setInteractionMode('symmetry');
      setAtomMode('symmetry');
      showNotify("Reset to defaults (aCVO)", "info");
    }
  }

  const handleImport = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const doc = yaml.load(event.target.result)
        // RESET: Start with default clean config
        const newConfig = JSON.parse(JSON.stringify(DEFAULT_CONFIG))

        // RESET derived states
        setNeighborDistances([])
        setBonds([])
        setPreviewAtoms([])
        setSelectedBondIdxs({})
        setCalcResults(null)
        setCalcError(null)
        setLogs([])

        // 1. Crystal Structure & Atoms (Process first to build label map)
        let labelMap = {}
        if (doc.crystal_structure) {
          if (doc.crystal_structure.lattice_parameters) {
            newConfig.lattice = { ...newConfig.lattice, ...doc.crystal_structure.lattice_parameters }
          }
          if (doc.crystal_structure.lattice_vectors) {
            const lv = doc.crystal_structure.lattice_vectors
            newConfig.lattice.lattice_vectors = lv
            // Derive a/b/c/angles from the vectors so the Lattice Constants
            // panel shows the real cell instead of the UI defaults (the 3D
            // view uses the vectors directly).
            if (!doc.crystal_structure.lattice_parameters &&
                Array.isArray(lv) && lv.length === 3) {
              const nrm = v => Math.hypot(v[0], v[1], v[2])
              const dot = (u, v) => u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
              const ang = (u, v) => {
                const d = nrm(u) * nrm(v)
                return d ? Math.acos(Math.max(-1, Math.min(1, dot(u, v)/d))) * 180/Math.PI : 90
              }
              const r5 = x => Number(x.toFixed(5))
              newConfig.lattice = {
                ...newConfig.lattice,
                a: r5(nrm(lv[0])), b: r5(nrm(lv[1])), c: r5(nrm(lv[2])),
                alpha: r5(ang(lv[1], lv[2])), beta: r5(ang(lv[0], lv[2])), gamma: r5(ang(lv[0], lv[1])),
                lattice_vectors: lv,
              }
            }
          }


          let atomsSource = null
          if (doc.crystal_structure.wyckoff_atoms) {
            atomsSource = doc.crystal_structure.wyckoff_atoms
            setAtomMode('symmetry')
            atomsSource.forEach((a, i) => labelMap[a.label || 'Atom'] = i)
          } else if (doc.crystal_structure.atoms_uc) {
            atomsSource = doc.crystal_structure.atoms_uc
            setAtomMode('explicit')
            atomsSource.forEach((a, i) => labelMap[a.label || 'Atom'] = i)
          }

          if (atomsSource) {
            newConfig.wyckoff_atoms = atomsSource.map(a => ({
              label: a.label || 'Atom',
              pos: a.pos || [0, 0, 0],
              spin_S: a.spin_S !== undefined ? a.spin_S : 0.5
            }))
          }
          if (doc.crystal_structure.magnetic_elements) {
            newConfig.magnetic_elements = doc.crystal_structure.magnetic_elements
          } else if (atomsSource) {
            const uniqueLabels = [...new Set(atomsSource.map(a => (a.label || a.species || '').replace(/[0-9]+$/, '')))].filter(x => x)
            if (uniqueLabels.length > 0) newConfig.magnetic_elements = uniqueLabels
          }
        }

        // 2. Interactions (Normalize pair -> atom_i/j)
        if (Array.isArray(doc.interactions)) {
          newConfig.explicit_interactions = doc.interactions.map(inter => {
            if (inter.pair && inter.atom_i === undefined) {
              const idxI = labelMap[inter.pair[0]]
              const idxJ = labelMap[inter.pair[1]]
              if (idxI !== undefined && idxJ !== undefined) {
                return { ...inter, atom_i: idxI, atom_j: idxJ }
              }
            }
            return inter
          })
          setInteractionMode('explicit')
        } else if (doc.interactions && doc.interactions.list) {
          newConfig.explicit_interactions = doc.interactions.list.map(inter => {
            if (inter.pair && inter.atom_i === undefined) {
              const idxI = labelMap[inter.pair[0]]
              const idxJ = labelMap[inter.pair[1]]
              if (idxI !== undefined && idxJ !== undefined) {
                return { ...inter, atom_i: idxI, atom_j: idxJ }
              }
            }
            return inter
          })
          setInteractionMode('explicit')
        } else if (doc.interactions && doc.interactions.symmetry_rules) {
          newConfig.symmetry_interactions = doc.interactions.symmetry_rules
          setInteractionMode('symmetry')
        }

        // 3. Other sections
        if (doc.parameters) {
          console.log("Importing parameters:", doc.parameters)
          // alert(JSON.stringify(doc.parameters)) 
          newConfig.parameters = { ...newConfig.parameters, ...doc.parameters }
        }
        if (doc.tasks) {
          const t = doc.tasks;
          newConfig.tasks = {
            ...newConfig.tasks,
            minimization: t.minimization ?? t.run_minimization ?? newConfig.tasks.minimization,
            dispersion: t.dispersion ?? t.run_dispersion ?? newConfig.tasks.dispersion,
            sqw_map: t.sqw_map ?? t.run_sqw_map ?? newConfig.tasks.sqw_map,
            powder_average: t.powder_average ?? t.run_powder_average ?? newConfig.tasks.powder_average,
            export_csv: t.export_csv ?? newConfig.tasks.export_csv,
            plot_dispersion: t.plot_dispersion ?? newConfig.tasks.plot_dispersion,
            plot_sqw_map: t.plot_sqw_map ?? newConfig.tasks.plot_sqw_map,
            plot_structure: t.plot_structure ?? newConfig.tasks.plot_structure,
            corrections: t.corrections ?? newConfig.tasks.corrections,
            scga: t.scga ?? newConfig.tasks.scga,
            thermal_mc: t.thermal_mc ?? newConfig.tasks.thermal_mc,
            sampled_correlations: t.sampled_correlations ?? newConfig.tasks.sampled_correlations,
            kpm_sqw: t.kpm_sqw ?? newConfig.tasks.kpm_sqw
          };
        }
        if (doc.plotting) newConfig.plotting = { ...newConfig.plotting, ...doc.plotting }
        if (doc.minimization) {
          newConfig.minimization = { ...newConfig.minimization, ...doc.minimization }
        }
        if (doc.powder_average) {
          newConfig.powder_average = { ...newConfig.powder_average, ...doc.powder_average }
        } else if (newConfig.tasks.powder_average && !newConfig.powder_average) {
          // Ensure it exists if the task is enabled
          newConfig.powder_average = { q_min: 0.1, q_max: 4.0, q_count: 50, num_samples: 50 }
        }
        if (doc.calculation) newConfig.calculation = { ...newConfig.calculation, ...doc.calculation }
        // Top-level `units:` (entangled mode) -> the UI field, so the run payload
        // re-emits it. Without this, imported dimer configs (e.g. Cu5SbO6) failed
        // with "entangled mode needs a `units:` list" from the web app.
        if (doc.units) {
          newConfig.calculation = { ...newConfig.calculation, units_text: JSON.stringify(doc.units) }
        }
        const joinList = (v) => Array.isArray(v) ? v.join(', ') : v
        if (doc.scga) newConfig.scga = { ...newConfig.scga, ...doc.scga }
        if (doc.thermal_mc) newConfig.thermal_mc = {
          ...newConfig.thermal_mc, ...doc.thermal_mc,
          temperatures: joinList(doc.thermal_mc.temperatures) ?? newConfig.thermal_mc.temperatures,
          supercell: joinList(doc.thermal_mc.supercell) ?? newConfig.thermal_mc.supercell
        }
        if (doc.sampled_correlations) newConfig.sampled_correlations = {
          ...newConfig.sampled_correlations, ...doc.sampled_correlations,
          supercell: joinList(doc.sampled_correlations.supercell) ?? newConfig.sampled_correlations.supercell
        }
        if (doc.kpm) newConfig.kpm = { ...newConfig.kpm, ...doc.kpm }
        if (doc.magnetic_structure) newConfig.magnetic_structure = { ...newConfig.magnetic_structure, ...doc.magnetic_structure }
        if (doc.q_path) {
          const { path, points_per_segment, ...points } = doc.q_path
          newConfig.q_path = {
            points: points || {},
            path: path || [],
            points_per_segment: points_per_segment || 100
          }
        }
        // Keep the raw structure/interactions/magnetic_structure so they are
        // sent to the backend verbatim (the designer model cannot represent
        // lattice_vectors, interaction_matrix, SIA, spiral orders, etc.).
        rawImportRef.current = doc.crystal_structure ? {
          crystal_structure: doc.crystal_structure,
          interactions: doc.interactions,
          magnetic_structure: doc.magnetic_structure,
          parameter_order: doc.parameter_order,
          // blocks with no UI editor yet: forwarded verbatim to the runner
          corrections: doc.corrections,
          energy_cut: doc.energy_cut,
        } : null
        setConfig(newConfig)
        alert('Configuration imported successfully! Previous state cleared.')
      } catch (err) {
        alert('Error parsing YAML: ' + err.message)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const handleExportYaml = async () => {
    // 1. Clean up and Re-order parameters for export
    const rawParams = JSON.parse(JSON.stringify(config.parameters));
    delete rawParams.S; // Remove misleading global S

    // Separate interactions from field parameters
    const fieldKeys = ['H_mag', 'H_dir'];
    const interactionKeys = Object.keys(rawParams).filter(k => !fieldKeys.includes(k)).sort();
    const sortedParamKeys = [...interactionKeys, ...fieldKeys].filter(k => rawParams[k] !== undefined);

    const cleanParams = {};
    sortedParamKeys.forEach(key => {
      let val = rawParams[key];
      if (typeof val === 'number') val = Number(val.toFixed(5));
      else if (Array.isArray(val)) val = val.map(v => typeof v === 'number' ? Number(v.toFixed(5)) : v);
      cleanParams[key] = val;
    });

    // 2. Clean up lattice and atoms
    const cleanLattice = { ...config.lattice };
    ['a', 'b', 'c', 'alpha', 'beta', 'gamma'].forEach(k => {
      if (typeof cleanLattice[k] === 'number') cleanLattice[k] = Number(cleanLattice[k].toFixed(5));
    });

    const cleanAtoms = config.wyckoff_atoms.map(a => ({
      ...a,
      pos: a.pos.map(v => Number(v.toFixed(5))),
      spin_S: typeof a.spin_S === 'number' ? Number(a.spin_S.toFixed(5)) : a.spin_S
    }));

    // 3. Structure the input for Export
    const sp = buildStructPayload()
    let expanded = {
      parameter_order: sp.parameter_order || sortedParamKeys,
      parameters: cleanParams,
      crystal_structure: rawImportRef.current
        ? sp.crystal_structure
        : {
            lattice_parameters: cleanLattice,
            wyckoff_atoms: cleanAtoms,
            atom_mode: atomMode,
            magnetic_elements: config.magnetic_elements || ["Cu"],
            dimensionality: 3
          },
      interactions: sp.interactions,
      magnetic_structure: sp.magnetic_structure,
      tasks: {
        ...config.tasks,
        calculate_dispersion: config.tasks.dispersion,
        calculate_sqw_map: config.tasks.sqw_map
      },
      q_path: {
        ...config.q_path.points,
        path: config.q_path.path,
        points_per_segment: config.q_path.points_per_segment
      },
      plotting: {
        ...config.plotting,
        energy_limits_disp: [config.plotting.energy_min, config.plotting.energy_max],
        broadening_width: config.plotting.broadening
      },
      minimization: {
        enabled: config.tasks.minimization,
        ...config.minimization
      },
      calculation: buildCalcPayload(),
      output: config.output,
      ...buildBeyondLswtBlocks()
    }

    try {
      if (!config.magnetic_structure.enabled) {
        delete expanded.magnetic_structure;
      }

      console.log('Expansion successful, generating file...')
      let yamlStr = yaml.dump(expanded)

      // Post-process to make vectors inline [x, y, z] via simple regex
      // Matches key followed by indented list of items
      const collapseVectors = (str) => {
        // Regex to match a key and a list of 2-8 items
        // Note: JS regex multiline mode.
        // We iterate through the string or use a robust pattern.
        // Pattern: (indent)(key):\n(indent+2)- val\n...

        const lines = str.split('\n');
        const newLines = [];
        let i = 0;

        while (i < lines.length) {
          const line = lines[i];
          const keyMatch = line.match(/^(\s*)([\w\d_]+):\s*$/);
          if (keyMatch) {
            const indent = keyMatch[1];
            const key = keyMatch[2];
            const items = [];
            let j = i + 1;
            let valid = true;

            // Collect list items
            while (j < lines.length) {
              const next = lines[j];
              const itemMatch = next.match(new RegExp(`^${indent}  - (.+)$`));
              if (!itemMatch) break;

              const val = itemMatch[1].trim();
              // Avoid nested objects
              if (val.includes(':') && !val.match(/^['"].*['"]$/)) {
                valid = false; break;
              }
              items.push(val);
              j++;
            }

            if (valid && items.length >= 2 && items.length <= 8) {
              newLines.push(`${indent}${key}: [${items.join(', ')}]`);
              i = j;
              continue;
            }
          }
          newLines.push(line);
          i++;
        }
        return newLines.join('\n');
      }

      const data = collapseVectors(yamlStr)

      if ('showSaveFilePicker' in window) {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: 'config_designer.yaml',
            types: [{
              description: 'YAML Configuration',
              accept: { 'text/yaml': ['.yaml', '.yml'] },
            }],
          });
          const writable = await handle.createWritable();
          await writable.write(data);
          await writable.close();
          showNotify(`Success! Configuration exported.`)
        } catch (err) {
          if (err.name !== 'AbortError') throw err;
        }
      } else {
        // Fallback for browsers without showSaveFilePicker
        const blob = new Blob([data], { type: 'text/yaml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'config_designer.yaml';
        link.click();
        showNotify(`Configuration exported (fallback download).`)
      }
    } catch (err) {
      console.error('Export error:', err)
      showNotify('Export failed: ' + err.message, 'error')
    }
  }


  // Symmetry Analysis State
  const [showSymmetryModal, setShowSymmetryModal] = useState(false)
  const [bondOrbits, setBondOrbits] = useState([])
  const [selectedOrbit, setSelectedOrbit] = useState(null)
  const [orbitConstraints, setOrbitConstraints] = useState(null)

  const fetchBondOrbits = async () => {
    try {
      showNotify("Analyzing bond symmetry...", "info")
      const response = await fetch('/api/analyze-bonds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          max_distance: 10.0,
          data: {
            crystal_structure: {
              lattice_parameters: config.lattice,
              wyckoff_atoms: config.wyckoff_atoms,
              atom_mode: atomMode
            }
          }
        }),
      })
      if (!response.ok) throw new Error('Failed to analyze bonds')
      const data = await response.json()
      setBondOrbits(data)
      setShowSymmetryModal(true)
      showNotify(`Found ${data.length} bond orbits.`, "success")
    } catch (err) {
      console.error('Error analyzing bonds:', err)
      showNotify("Failed to analyze symmetry. Check server logs.", "error")
    }
  }

  const fetchConstraints = async (orbit) => {
    try {
      const response = await fetch('/api/bond-constraints', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bond: orbit.representative,
          data: {
            crystal_structure: {
              lattice_parameters: config.lattice,
              wyckoff_atoms: config.wyckoff_atoms,
              atom_mode: atomMode
            }
          }
        }),
      })
      if (!response.ok) throw new Error('Failed to get constraints')
      const data = await response.json()
      setOrbitConstraints(data)
      setSelectedOrbit(orbit)
    } catch (err) {
      console.error('Error fetching constraints:', err)
      showNotify("Failed to fetch bond constraints.", "error")
    }
  }

  const handleAddSymmetryInteraction = (orbit, constraints, params) => {
    // params is a dict of { "J1": 1.5, "D": 0.1 } etc.
    // We need to map this to the config.
    // Actually, for "interaction_matrix", we store the matrix components OR the free parameters?
    // The current backend for 'interaction_matrix' expects a 3x3 matrix in 'value'.
    // BUT, to keep it editable, maybe we should store the symbolic map?
    // Complexity: The backend 'add_symmetry_interaction' takes a numeric matrix or symbolic list-of-lists?
    // It supports 'interaction_matrix' type.

    // Let's construct the numeric matrix here for simplicity, OR pass the params if backend supported it.
    // The backend 'calc_interaction_matrix' is available but not exposed as endpoint yet.
    // Let's do a client-side substitution for the initial value.

    // better: The User inputs J1, D... we save them in 'value' as a special object?
    // No, let's stick to the schema: value is 3x3 array of strings/numbers.
    // unique strings for params.

    // 1. Construct the matrix string/value from constraints + params
    const matrix = constraints.symbolic_matrix.map(row => row.map(cell => {
      // If cell is '0' or '0.0', keep it.
      // If cell is a symbol (e.g. 'j0'), check if we have a mapped param name?
      // We need a mapping from symbolic vars (j0, j1) to User Params (J1, D...).
      // Creating this mapping is tricky without user input.
      // Let's just use the user-provided params directly if they match the symbolic slots.
      return cell; // Placeholder
    }));

    // Actually, simpler approach for V1:
    // Just add a "interaction_matrix" entry with the representative bond.
    // And initializes the 'value' with the symbolic matrix from constraints.
    // The user can then edit the values in the main UI (which needs update for matrix support).

    const newRule = {
      type: 'interaction_matrix',
      ref_pair: [orbit.representative.atom_i, orbit.representative.atom_j],
      offset: orbit.representative.offset,
      distance: Number(orbit.distance.toFixed(5)),
      value: constraints.symbolic_matrix, // 3x3 array of strings
      constraints: constraints // Store for reference/UI helpers?
    };

    // Update global parameters with free symbols (init to 0)
    const newParams = { ...config.parameters };
    constraints.free_parameters.forEach(p => {
      if (newParams[p] === undefined) newParams[p] = 0.0;
    });

    setConfig({
      ...config,
      symmetry_interactions: [...config.symmetry_interactions, newRule],
      parameters: newParams
    });
    setShowSymmetryModal(false);
    showNotify("Added Symmetry Matrix Interaction", "success");
  }

  const fetchNeighbors = async () => {
    try {
      const response = await fetch('/api/get-neighbors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: {
            crystal_structure: {
              lattice_parameters: config.lattice,
              wyckoff_atoms: config.wyckoff_atoms,
              dimensionality: config.lattice.dimensionality,
              atom_mode: atomMode
            }
          }
        }),
      })
      if (!response.ok) throw new Error('Failed to fetch neighbors')
      const data = await response.json()
      setNeighborDistances(data)
    } catch (err) {
      console.error('Error fetching neighbors:', err)
      showNotify("Failed to fetch neighbor suggestions. Check server logs.", "error")
    }
  }

  React.useEffect(() => {
    if (activeTab === 'interactions') {
      fetchNeighbors()
    }
  }, [activeTab, config.lattice, config.wyckoff_atoms])

  // ---- Beyond-LSWT payload helpers ------------------------------------- //
  // Parse '0.5, 1, 2' or '4,4,1' style strings into number lists.
  const parseNumList = (v, fallback) => {
    if (Array.isArray(v)) return v.map(Number)
    if (typeof v !== 'string') return fallback
    const out = v.split(/[\s,]+/).filter(Boolean).map(Number).filter(x => !isNaN(x))
    return out.length ? out : fallback
  }

  // The config blocks for the beyond-LSWT tasks (SCGA, thermal MC, classical
  // dynamics, KPM) plus entangled-mode extras. Only enabled tasks emit a block, so
  // the YAML stays clean. Consumed by the runner's task blocks (see TUTORIAL 4h).
  const buildBeyondLswtBlocks = () => {
    const out = {}
    if (config.tasks.scga) out.scga = {
      temperature: Number(config.scga.temperature) || 1.0,
      mesh_density: Number(config.scga.mesh_density) || 12
    }
    if (config.tasks.thermal_mc) out.thermal_mc = {
      temperatures: parseNumList(config.thermal_mc.temperatures, [0.5, 1, 2, 4]),
      supercell: parseNumList(config.thermal_mc.supercell, [4, 4, 1]),
      n_sweeps: Number(config.thermal_mc.n_sweeps) || 4000,
      n_equil: Number(config.thermal_mc.n_equil) || 1500
    }
    if (config.tasks.sampled_correlations) out.sampled_correlations = {
      temperature: Number(config.sampled_correlations.temperature) || 0.5,
      supercell: parseNumList(config.sampled_correlations.supercell, [8, 1, 1]),
      dt: Number(config.sampled_correlations.dt) || 0.02,
      n_steps: Number(config.sampled_correlations.n_steps) || 2048,
      n_traj: Number(config.sampled_correlations.n_traj) || 8
    }
    if (config.tasks.kpm_sqw) out.kpm = {
      e_min: Number(config.kpm.e_min) || 0,
      e_max: Number(config.kpm.e_max) || 10,
      e_step: Number(config.kpm.e_step) || 0.05,
      fwhm: Number(config.kpm.fwhm) || 0.1
    }
    if (config.calculation.mode === 'entangled' && config.calculation.units_text) {
      try {
        const u = JSON.parse(config.calculation.units_text)
        if (Array.isArray(u) && u.length) out.units = u
      } catch { /* leave units for the runner to derive/complain about */ }
    }
    return out
  }

  // calculation block for the payload: strip UI-only fields; only send
  // series_order when the entangled series is actually requested.
  const buildCalcPayload = () => {
    const calc = { ...config.calculation }
    delete calc.units_text
    if (!(calc.mode === 'entangled' && Number(calc.series_order) > 0)) {
      delete calc.series_order
      delete calc.series_resum
    }
    return calc
  }

  const updateField = (section, field, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: { ...prev[section], [field]: value }
    }))
  }

  // Monte-Carlo methods ('anneal', 'steep') and the gradient multistart methods take
  // completely different budgets: `num_starts` means "annealing runs" (a handful) for
  // the former and "random restarts" (hundreds) for the latter, and early_stopping /
  // n_workers apply only to the latter. Carrying one method's numbers over to the
  // other is how you get either an absurdly slow run or a silently wrong ground state,
  // so retune them whenever the method changes.
  const isAnnealMethod = (m) => m === 'anneal' || m === 'monte_carlo' || m === 'steep'

  const onMinimizationMethodChange = (method) => {
    setConfig(prev => {
      const wasAnneal = isAnnealMethod(prev.minimization.method)
      const nowAnneal = isAnnealMethod(method)
      const next = { ...prev.minimization, method }
      if (wasAnneal !== nowAnneal) {
        if (nowAnneal) {
          next.num_starts = 4          // independent annealing runs
          next.n_sweeps = next.n_sweeps ?? 2000
        } else {
          next.num_starts = 1000       // random restarts
          next.early_stopping = 10
        }
      }
      return { ...prev, minimization: next }
    })
  }

  const addRuleFromVisualizer = (type) => {
    if (!selectedBond) return;

    // Construct new rule
    // We need 'ref_pair', 'offset', 'distance', 'value'
    // Visualizer bond object: { atom_i, atom_j, offset, distance, ... }
    // Note: distance might need to be calculated if not present, but usually backend gives it? 
    // Actually Visualizer calculates display distance. 
    // Let's assume we can compute or it's there. 
    // If 'distance' is missing in bond object, we can approximate it or re-fetch?
    // Let's look at `Visualizer.jsx`: it uses start/end to get distance.
    // The `bond` object from `get-visualizer-data` usually mimics the input interaction OR neighbor list.
    // If it comes from 'bonds' list, it might NOT have distance.
    // Safeguard: Use a default distance or calculate from config.lattice?
    // Better: Just use 0.0 or prompt? Or try to find it in neighbor list?

    // Simplest: Add it with a placeholder distance if missing, user can adjust.
    // But let's try to find it in neighborDistances if available?

    const getInitValue = (t) => {
      if (t === 'heisenberg') return 'J0';
      if (t === 'kitaev') return 'K1';
      if (t === 'interaction_matrix') return [['0', '0', '0'], ['0', '0', '0'], ['0', '0', '0']];
      if (t === 'dm') return ['D1', 'D2', 'D3'];
      return ['G1', 'G2', 'G3']; // anisotropic_exchange
    };

    const newRule = {
      type: type,
      ref_pair: [previewAtoms[selectedBond.atom_i]?.label || "?", previewAtoms[selectedBond.atom_j]?.label || "?"],
      offset: selectedBond.offset || [0, 0, 0],
      distance: Number((selectedBond.distance || 0.0).toFixed(5)),
      value: getInitValue(type)
    }

    // Fallback for explicit mode or if labels missing
    // For explicit interactions, we use indices in 'atom_i', 'atom_j'
    if (interactionMode === 'explicit') {
      delete newRule.ref_pair;
      newRule.atom_i = selectedBond.atom_i;
      newRule.atom_j = selectedBond.atom_j;
      newRule.offset_j = selectedBond.offset || [0, 0, 0];
    }

    // Add to config
    if (interactionMode === 'symmetry') {
      setConfig(prev => ({
        ...prev,
        symmetry_interactions: [...prev.symmetry_interactions, newRule]
      }));
    } else {
      setConfig(prev => ({
        ...prev,
        explicit_interactions: [...prev.explicit_interactions, {
          type,
          atom_i: selectedBond.atom_i,
          atom_j: selectedBond.atom_j,
          offset_j: selectedBond.offset || [0, 0, 0],
          distance: selectedBond.distance || 0.0,
          value: getInitValue(type)
        }]
      }));
    }
    showNotify(`Added ${type} interaction`, 'success');
  }

  const stopCalculation = async () => {
    if (!calcLoading || calcStopping) return
    setCalcStopping(true)
    stopRequestedRef.current = true
    try {
      await fetch('/api/stop-calculation', { method: 'POST' })
      showNotify("Stopping calculation...", "info")
    } catch (err) {
      console.error(err)
      showNotify("Failed to stop calculation", "error")
    }
  }

  const runCalculation = async (overrides = {}) => {
    setCalcLoading(true)
    setCalcStopping(false)
    stopRequestedRef.current = false
    setCalcError(null)
    setCalcResults(null)
    setJsonCache({}) // Clear 3D structure data cache
    setLogs([]) // Clear previous logs

    // Construct payload as expected by expand-config logic backend
    const sp = buildStructPayload()
    const input = {
      crystal_structure: sp.crystal_structure,
      interactions: sp.interactions,
      magnetic_structure: sp.magnetic_structure,
      parameter_order: sp.parameter_order,
      parameters: config.parameters,
      tasks: config.tasks,
      q_path: {
        ...config.q_path.points,
        path: config.q_path.path,
        points_per_segment: config.q_path.points_per_segment
      },
      plotting: {
        ...config.plotting,
        energy_limits_disp: [config.plotting.energy_min, config.plotting.energy_max],
        broadening_width: config.plotting.broadening
      },
      minimization: {
        enabled: config.tasks.minimization,
        ...config.minimization
      },
      powder_average: config.powder_average,
      calculation: buildCalcPayload(),
      output: config.output,
      fitting: config.fitting,
      ...buildBeyondLswtBlocks()
    }
    if (rawImportRef.current?.corrections) input.corrections = rawImportRef.current.corrections
    if (rawImportRef.current?.energy_cut) input.energy_cut = rawImportRef.current.energy_cut

    // Per-run overrides (e.g. the Fitting panel forces `tasks: { fit: true }`
    // so only the fit runs, regardless of the Tasks panel's checkboxes).
    if (overrides.tasks) input.tasks = overrides.tasks

    try {
      // First ensure backend can expand it (optional check, but good for robust config)
      // Actually, run-calculation endpoint expects the raw data structure
      const response = await fetch('/api/run-calculation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: input }),
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Calculation failed')
      }

      const data = await response.json()
      setCalcResults(data)
      showNotify("Calculation completed!", "success")
    } catch (err) {
      console.error(err)
      if (stopRequestedRef.current) {
        // User intentionally stopped — don't surface it as a failure.
        showNotify("Calculation stopped", "info")
      } else {
        setCalcError(err.message)
        showNotify("Calculation failed", "error")
      }
    } finally {
      setCalcLoading(false)
      setCalcStopping(false)
      stopRequestedRef.current = false
    }
  }

  // --- Data fitting helpers ---------------------------------------------- //
  const updateFitting = (patch) =>
    setConfig(c => ({ ...c, fitting: { ...c.fitting, ...patch } }))

  const toggleVaryParam = (name) => {
    setConfig(c => {
      const vary = new Set(c.fitting.vary || [])
      if (vary.has(name)) vary.delete(name); else vary.add(name)
      return { ...c, fitting: { ...c.fitting, vary: Array.from(vary) } }
    })
  }

  const setParamBound = (name, idx, value) => {
    setConfig(c => {
      const bounds = { ...(c.fitting.bounds || {}) }
      const cur = bounds[name] ? [...bounds[name]] : [null, null]
      cur[idx] = value === '' ? null : parseFloat(value)
      bounds[name] = cur
      return { ...c, fitting: { ...c.fitting, bounds } }
    })
  }

  const handleFitDataUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch('/api/upload-fit-data', { method: 'POST', body: fd })
      if (!res.ok) throw new Error('Upload failed')
      const data = await res.json()
      updateFitting({ data_file: data.data_file, data_label: data.original_name })
      showNotify(`Loaded data: ${data.original_name}`, 'success')
    } catch (err) {
      showNotify('Failed to upload data file', 'error')
    }
  }

  const runFit = async () => {
    if (!config.fitting.data_file) {
      showNotify('Upload an experimental data file first', 'error')
      return
    }
    if (!(config.fitting.vary || []).length) {
      showNotify('Select at least one parameter to vary', 'error')
      return
    }
    setActiveTab('run')
    // Only run the fit (+ comparison plot); ignore the Tasks panel's switches.
    await runCalculation({ tasks: { fit: true, plot_fit: true } })
  }

  // Scalar parameter names available to vary (vectors like H_dir are excluded).
  const fittableParams = Object.entries(config.parameters || {})
    .filter(([k, v]) => k !== 'S' && !Array.isArray(v))
    .map(([k]) => k)

  // Import a minimized magnetic structure (from the energy-minimization result)
  // into the Manual Magnetic Structure tab as a per-spin 'generic' direction
  // list, so it can be reused as a fixed input (for plain runs or fitting)
  // without re-running minimization. Minimization is disabled so subsequent runs
  // use the imported structure directly.
  const importMinimizedStructure = (data) => {
    const vectors = (data && data.vectors) || []
    if (!vectors.length) {
      showNotify('No structure vectors found to import', 'error')
      return
    }
    // Keep FULL precision: the LSWT spin-wave intensities (S(Q,ω)) are evaluated
    // about the minimized ground state, and the Bogoliubov eigenvectors are
    // sensitive to small deviations from it. Rounding the directions (e.g. to 5
    // decimals) nudges the structure off the true minimum — energies stay
    // correct (the energy is stationary there) but S(Q,ω) intensities shift.
    const directions = vectors.map(v => v.map(x => Number(x)))
    setConfig(c => ({
      ...c,
      magnetic_structure: {
        ...c.magnetic_structure,
        enabled: true,
        type: 'pattern',
        pattern_type: 'generic',
        directions
      },
      tasks: { ...c.tasks, minimization: false }
    }))
    showNotify(`Imported ${directions.length} spins into Manual Structure (minimization disabled)`, 'success')
    setActiveTab('magstruct')
  }

  return (
    <div className="app-container">
      <div className="background-glow"></div>

      <AppHeader
        onCifUpload={handleCifUpload}
        onMcifUpload={handleMcifUpload}
        onYamlImport={handleImport}
        onReset={resetToDefaults}
        onExportYaml={handleExportYaml}
      />

      <main>
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} width={sidebarWidth} />

        <div
          className="resizer"
          onMouseDown={startResizing('left')}
        ></div>

        {activeTab !== 'run' && (
          <section className="content-area animate-fade-in">
            {activeTab === 'structure' && (
              <div className="form-section">
                <h2 className="section-title mb-xl">Crystal Architecture</h2>
                <div className="card shadow-glow">
                  <div className="mb-lg">
                    <h3 className="mb-md text-xs opacity-60 tracking-wider">Lattice Constants (Å)</h3>
                    <div className="lattice-grid">
                      {['a', 'b', 'c'].map(k => (
                        <div key={k} className="input-group">
                          <label>{k}</label>
                          <input type="number" step="0.001" value={config.lattice[k]} className="minimal-input"
                            onChange={(e) => updateField('lattice', k, parseFloat(e.target.value))} />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mb-lg">
                    <h3 className="mb-md text-xs opacity-60 tracking-wider">Angles (°)</h3>
                    <div className="lattice-grid">
                      {['alpha', 'beta', 'gamma'].map(k => (
                        <div key={k} className="input-group">
                          <label>{k}</label>
                          <input
                            type="number"
                            step="0.001"
                            value={config.lattice[k]}
                            className={`minimal-input ${(config.lattice.dimensionality === '2D' && (k === 'alpha' || k === 'beta')) ? 'opacity-40 pointer-events-none' : ''}`}
                            disabled={config.lattice.dimensionality === '2D' && (k === 'alpha' || k === 'beta')}
                            onChange={(e) => updateField('lattice', k, parseFloat(e.target.value))}
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid-form border-t border-light pt-lg mt-md">
                    <div className="input-group">
                      <label>SPACE GROUP</label>
                      <div className="relative" ref={sgDropdownRef}>
                        <div
                          className="minimal-input flex-between cursor-pointer"
                          onClick={() => setIsSgDropdownOpen(!isSgDropdownOpen)}
                        >
                          <span>
                            {config.lattice.space_group ? (
                              <>
                                <span className="text-muted mr-xs">No. {config.lattice.space_group}</span>
                                <span className="font-medium text-text">
                                  {spaceGroupsList.find(s => s.number === config.lattice.space_group)?.symbol || ""}
                                </span>
                              </>
                            ) : <span className="text-muted">Select Space Group...</span>}
                          </span>
                          <ChevronDown size={14} className={`transition-transform ${isSgDropdownOpen ? 'rotate-180' : ''}`} />
                        </div>

                        {isSgDropdownOpen && (
                          <div className="absolute top-full left-0 w-full mt-1 bg-surface-2 border border-border rounded-lg shadow-xl z-50 max-h-60 flex flex-col animate-fade-in-up">
                            <div className="p-2 border-b border-border sticky top-0 bg-surface-2">
                              <div className="relative">
                                <Search size={14} className="absolute left-2 top-2.5 text-muted" />
                                <input
                                  type="text"
                                  className="sg-search-input"
                                  placeholder="Search (e.g. 227 or Fd-3m)..."
                                  value={sgSearch}
                                  onChange={(e) => setSgSearch(e.target.value)}
                                  autoFocus
                                  onClick={(e) => e.stopPropagation()}
                                />
                              </div>
                            </div>
                            <div className="overflow-y-auto flex-1 p-1 custom-scrollbar">
                              {spaceGroupsList.filter(sg =>
                                sg.number.toString().includes(sgSearch) ||
                                sg.symbol.toLowerCase().includes(sgSearch.toLowerCase())
                              ).map(sg => (
                                <div
                                  key={sg.number}
                                  className={`sg-dropdown-item ${config.lattice.space_group === sg.number ? 'selected' : ''}`}
                                  onClick={() => {
                                    updateField('lattice', 'space_group', sg.number);
                                    setIsSgDropdownOpen(false);
                                    setSgSearch("");
                                  }}
                                >
                                  <span className="sg-number">{sg.number}</span>
                                  <span className="sg-symbol">{sg.symbol}</span>
                                  {config.lattice.space_group === sg.number && <Check size={14} className="text-accent" />}
                                </div>
                              ))}
                              {spaceGroupsList.filter(sg => sg.number.toString().includes(sgSearch) || sg.symbol.toLowerCase().includes(sgSearch.toLowerCase())).length === 0 && (
                                <div className="p-4 text-center text-xs text-muted">No space groups found.</div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                </div>

                <div className="card mt-xl">
                  <div className="flex-between mb-md align-end">
                    <div>
                      <h3 className="mb-xs">Basis Atoms</h3>
                      <div className="modern-toggle-group mb-sm">
                        <button
                          className={`toggle-btn ${atomMode === 'symmetry' ? 'active' : ''}`}
                          onClick={() => setAtomMode('symmetry')}
                        >
                          <Wind size={14} className="mr-xs" />
                          Wyckoff Positions
                        </button>
                        <button
                          className={`toggle-btn ${atomMode === 'explicit' ? 'active' : ''}`}
                          onClick={() => setAtomMode('explicit')}
                        >
                          <Box size={14} className="mr-xs" />
                          Explicit Unit Cell
                        </button>
                      </div>
                      <p className="text-xs text-muted max-w-md">
                        {atomMode === 'symmetry'
                          ? "Define unique atoms (Wyckoff positions). The full structure will be generated using the Space Group symmetry."
                          : "Define all atoms in the unit cell explicitly. Space group symmetry will be ignored for atomic positions."}
                      </p>
                    </div>
                    <button className="btn btn-secondary btn-sm" onClick={() => {
                      const next = [...config.wyckoff_atoms];
                      next.push({ label: 'Cu', pos: [0, 0, 0], spin_S: 0.5 });
                      setConfig({ ...config, wyckoff_atoms: next });
                    }}><Plus size={14} /> Add Atom</button>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Label</th>
                        <th>Pos (x,y,z)</th>
                        <th>S</th>
                        <th>Ion/Element</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {config.wyckoff_atoms.map((atom, idx) => (
                        <tr key={idx}>
                          <td><input type="text" className="table-input mono" value={atom.label} onChange={(e) => {
                            const next = [...config.wyckoff_atoms]; next[idx].label = e.target.value; setConfig({ ...config, wyckoff_atoms: next })
                          }} /></td>
                          <td>
                            <div className="flex-gap-xs">
                              {[0, 1, 2].map(i => (
                                <input key={i} type="number" step="0.01" value={atom.pos[i]} className="table-input" onChange={(e) => {
                                  const next = [...config.wyckoff_atoms]; next[idx].pos[i] = parseFloat(e.target.value); setConfig({ ...config, wyckoff_atoms: next })
                                }} />
                              ))}
                            </div>
                          </td>
                          <td>
                            <input type="number" step="0.5" className="minimal-input" value={atom.spin_S} onChange={(e) => {
                              const next = [...config.wyckoff_atoms]
                              next[idx].spin_S = parseFloat(e.target.value)
                              setConfig({ ...config, wyckoff_atoms: next })
                            }} />
                          </td>
                          <td>
                            <input type="text" className="table-input mono" placeholder="e.g. Fe3+" value={atom.ion || ''} onChange={(e) => {
                              const next = [...config.wyckoff_atoms]; next[idx].ion = e.target.value; setConfig({ ...config, wyckoff_atoms: next })
                            }} />
                          </td>
                          <td><button onClick={() => {
                            const next = config.wyckoff_atoms.filter((_, i) => i !== idx); setConfig({ ...config, wyckoff_atoms: next })
                          }} className="icon-btn text-error"><Trash2 size={14} /></button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {activeTab === 'interactions' && (
              <div className="form-section">


                {interactionMode === 'symmetry' ? (
                  <>
                    <h2 className="section-title compact mb-xl">Bonding Rules</h2>

                    <div className="flex align-center gap-md mb-lg">
                      <div className="modern-toggle-group">
                        <button
                          className={`toggle-btn ${interactionMode === 'symmetry' ? 'active' : ''}`}
                          onClick={() => setInteractionMode('symmetry')}
                        >
                          <Wind size={14} className="mr-xs" />
                          Symmetry Rules
                        </button>
                        <button
                          className={`toggle-btn ${interactionMode === 'explicit' ? 'active' : ''}`}
                          onClick={() => setInteractionMode('explicit')}
                        >
                          <Activity size={14} className="mr-xs" />
                          Explicit Interactions
                        </button>
                      </div>

                      <button className="btn btn-primary btn-sm" onClick={() => {
                        const nextRules = [...config.symmetry_interactions, { type: 'heisenberg', distance: 3.0, value: 'J1' }];
                        const nextParams = { ...config.parameters };
                        if (!nextParams.J1) nextParams.J1 = 0.0;
                        setConfig({ ...config, symmetry_interactions: nextRules, parameters: nextParams });
                      }}><Plus size={14} /> Add Rule</button>
                    </div>
                    <div className="interaction-grid">
                      {config.symmetry_interactions.map((inter, idx) => (
                        <div key={idx} className="interaction-card animate-fade-in">
                          <div className="interaction-header">
                            <div className="interaction-info">
                              <div className="interaction-icon-box">
                                {inter.type === 'heisenberg' ? <Zap size={16} /> : (inter.type === 'dm' ? <Wind size={16} /> : <Crosshair size={16} />)}
                              </div>
                              <div>
                                <span className="interaction-type">{(inter.type === 'dm' || inter.type === 'dm_interaction') ? 'Dzyaloshinskii–Moriya' : inter.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                                <span className="interaction-subtitle">
                                  {inter.ref_pair ? `Ref: ${inter.ref_pair.join('-')}` : 'Auto-detected'}
                                  {inter.offset && (inter.offset[0] !== 0 || inter.offset[1] !== 0 || inter.offset[2] !== 0) && ` [${inter.offset.join(',')}]`}
                                </span>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <button
                                className={`icon-btn ${hiddenBondLabels.has(getBondKey(inter.value)) ? 'opacity-50' : 'text-primary'}`}
                                onClick={() => toggleBondVisibility(inter.value)}
                                title={hiddenBondLabels.has(getBondKey(inter.value)) ? "Show bonds" : "Hide bonds"}
                              >
                                {hiddenBondLabels.has(getBondKey(inter.value)) ? <EyeOff size={14} /> : <Eye size={14} />}
                              </button>
                              <button onClick={() => {
                                const next = config.symmetry_interactions.filter((_, i) => i !== idx); setConfig({ ...config, symmetry_interactions: next })
                              }} className="icon-btn text-error"><Trash2 size={14} /></button>
                            </div>
                          </div>

                          <div className="interaction-params">
                            <div className="input-group">
                              <label>Distance (Å)</label>
                              <input type="number" step="0.01" className="minimal-input" value={typeof inter.distance === 'number' ? Number(inter.distance.toFixed(5)) : inter.distance} onChange={(e) => {
                                const next = [...config.symmetry_interactions]; next[idx].distance = parseFloat(e.target.value); setConfig({ ...config, symmetry_interactions: next })
                              }} />
                            </div>
                            <div className="input-group">
                              <label>{inter.type === 'kitaev' ? 'Coupling (K)' : 'Value'}</label>
                              {inter.type === 'kitaev' ? (
                                <div className="flex gap-2">
                                  <input type="text" className="minimal-input accent-text flex-1" value={inter.value || inter.K} onChange={(e) => {
                                    const next = [...config.symmetry_interactions]; next[idx].value = e.target.value; setConfig({ ...config, symmetry_interactions: next })
                                  }} />
                                  <select className="minimal-select w-16" value={inter.bond_direction || 'x'} onChange={(e) => {
                                    const next = [...config.symmetry_interactions]; next[idx].bond_direction = e.target.value; setConfig({ ...config, symmetry_interactions: next })
                                  }}>
                                    <option value="x">X</option>
                                    <option value="y">Y</option>
                                    <option value="z">Z</option>
                                  </select>
                                </div>
                              ) : (inter.type === 'interaction_matrix' && Array.isArray(inter.value)) ? (
                                <div
                                  className="bg-black/20 p-xs rounded border border-color/30"
                                  style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px' }}
                                >
                                  {inter.value.map((row, r) => row.map((cell, c) => (
                                    <input
                                      key={`${r}-${c}`}
                                      type="text"
                                      className={`text-center text-xs p-1 bg-transparent border-none outline-none w-full ${cell === '0' || cell === '0.0' ? 'opacity-30' : 'text-accent font-bold'}`}
                                      style={{ border: '1px solid rgba(255,255,255,0.05)', borderRadius: '4px' }}
                                      value={cell}
                                      onChange={(e) => {
                                        const next = [...config.symmetry_interactions];
                                        // Deep copy matrix
                                        const newMatrix = next[idx].value.map(row => [...row]);
                                        newMatrix[r][c] = e.target.value;
                                        next[idx].value = newMatrix;
                                        setConfig({ ...config, symmetry_interactions: next });
                                      }}
                                    />
                                  )))}
                                </div>
                              ) : (
                                <input type="text" className="minimal-input accent-text" value={inter.value} onChange={(e) => {
                                  const next = [...config.symmetry_interactions]; next[idx].value = e.target.value; setConfig({ ...config, symmetry_interactions: next })
                                }} />
                              )}
                            </div>
                            <div className="input-group">
                              <label>Type</label>
                              <select
                                className="minimal-select"
                                value={inter.type}
                                onChange={(e) => {
                                  const next = [...config.symmetry_interactions];
                                  next[idx].type = e.target.value;
                                  if (e.target.value === 'interaction_matrix') {
                                    next[idx].value = [['0', '0', '0'], ['0', '0', '0'], ['0', '0', '0']];
                                  } else if (e.target.value === 'heisenberg') {
                                    // Reset to scalar if switching back, to avoid confusion
                                    if (Array.isArray(next[idx].value)) next[idx].value = 'J1';
                                  }
                                  setConfig({ ...config, symmetry_interactions: next })
                                }}
                              >
                                <option value="heisenberg">Heisenberg</option>
                                <option value="dm">DM Interaction</option>
                                <option value="anisotropic_exchange">Anisotropic</option>
                                <option value="interaction_matrix">Interaction Matrix</option>
                                <option value="kitaev">Kitaev</option>
                              </select>
                            </div>
                          </div>

                          {/* Review Matrix Display */}
                          {(() => {
                            const matrix = calculateExchangeMatrixSymbolic(inter, config.parameters);
                            if (!matrix) return null;
                            return (
                              <div className="exchange-matrix-panel">
                                <div className="exchange-matrix-label">Exchange Tensor (J<sub>ij</sub>)</div>
                                <div className="exchange-matrix-grid">
                                  {matrix.flat().map((val, i) => (
                                    <div key={i} className={`exchange-matrix-cell ${val === 0 || val === '0' || val === '0.0' ? 'zero' : ''}`}>
                                      {typeof val === 'number' ? Number(val.toFixed(5)) : val}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                      ))}
                    </div>

                    <div className="mt-xl border-t border-color/30 pt-xl">
                      <div className="flex-between mb-lg">
                        <h2 className="section-title compact mb-0">Single-Ion Anisotropy</h2>
                        <button className="btn btn-secondary btn-sm" onClick={() => {
                          const next = [...(config.single_ion_anisotropy || []), { type: 'sia', atom_label: config.wyckoff_atoms[0]?.label || 'Cu', value: 'D', axis: [0, 0, 1] }];
                          const nextParams = { ...config.parameters };
                          if (!nextParams.D) nextParams.D = 0.0;
                          setConfig({ ...config, single_ion_anisotropy: next, parameters: nextParams });
                        }}><Plus size={14} /> Add SIA</button>
                      </div>

                      <div className="interaction-grid">
                        {(config.single_ion_anisotropy || []).map((sia, idx) => (
                          <div key={idx} className="interaction-card animate-fade-in">
                            <div className="interaction-header">
                              <div className="interaction-info">
                                <div className="interaction-icon-box">
                                  <Zap size={16} />
                                </div>
                                <div>
                                  <span className="interaction-type">Single-Ion Anisotropy</span>
                                  <span className="interaction-subtitle">Atom: {sia.atom_label}</span>
                                </div>
                              </div>
                              <button onClick={() => {
                                const next = config.single_ion_anisotropy.filter((_, i) => i !== idx);
                                setConfig({ ...config, single_ion_anisotropy: next })
                              }} className="icon-btn text-error"><Trash2 size={14} /></button>
                            </div>

                            <div className="interaction-params">
                              <div className="input-group">
                                <label>Atom Label</label>
                                <select className="minimal-select" value={sia.atom_label} onChange={(e) => {
                                  const next = [...config.single_ion_anisotropy]; next[idx].atom_label = e.target.value; setConfig({ ...config, single_ion_anisotropy: next })
                                }}>
                                  {config.wyckoff_atoms.map(a => <option key={a.label} value={a.label}>{a.label}</option>)}
                                </select>
                              </div>
                              <div className="input-group">
                                <label>K / D Constant</label>
                                <input type="text" className="minimal-input accent-text" value={sia.value} onChange={(e) => {
                                  const next = [...config.single_ion_anisotropy]; next[idx].value = e.target.value; setConfig({ ...config, single_ion_anisotropy: next })
                                }} />
                              </div>
                              <div className="input-group">
                                <label>Anisotropy Axis</label>
                                <div className="vector-input-grid">
                                  {[0, 1, 2].map(k => (
                                    <input key={k} type="number" step="0.1" className="table-input center" value={sia.axis[k]} onChange={(e) => {
                                      const next = [...config.single_ion_anisotropy];
                                      const nextAxis = [...next[idx].axis];
                                      nextAxis[k] = parseFloat(e.target.value);
                                      next[idx].axis = nextAxis;
                                      setConfig({ ...config, single_ion_anisotropy: next })
                                    }} />
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="mt-md border-t border-color/30 pt-xl">
                      <h3 className="section-title text-sm mb-sm">Symmetry Analysis</h3>
                      <div className="flex gap-2 mb-sm items-center">
                        <button className="btn btn-secondary btn-sm" onClick={fetchBondOrbits}>
                          <Activity size={14} className="mr-xs" />
                          Analyze Symmetry Orbits
                        </button>
                      </div>

                      {/* Symmetry Modal Overlay */}
                      {showSymmetryModal && (
                        <div className="fixed inset-0 bg-black/50 z-50 flex center animate-fade-in" onClick={() => setShowSymmetryModal(false)}>
                          <div className="bg-surface border border-color rounded-xl p-lg shadow-glow max-w-2xl w-full max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
                            <h3 className="text-xl font-bold mb-md flex-between">
                              <span>Bond Orbits</span>
                              <button className="icon-btn" onClick={() => setShowSymmetryModal(false)}><Crosshair size={20} /></button>
                            </h3>

                            {!selectedOrbit ? (
                              <div className="grid gap-sm">
                                {bondOrbits.map((orb, i) => (
                                  <div key={i} className="card hover-glow cursor-pointer flex-between" onClick={() => fetchConstraints(orb)}>
                                    <div>
                                      <div className="text-lg font-mono text-accent">{orb.distance.toFixed(4)} Å</div>
                                      <div className="text-xs text-muted">Multiplicity: {orb.multiplicity}</div>
                                    </div>
                                    <div className="text-right">
                                      <div className="text-sm font-bold">{orb.representative.atom_i} → {orb.representative.atom_j}</div>
                                      <div className="text-xs text-muted font-mono">Offset: [{orb.representative.offset.join(',')}]</div>
                                    </div>
                                    <ChevronRight className="opacity-50" />
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="animate-slide-in">
                                <button className="btn btn-xs btn-secondary mb-md" onClick={() => { setSelectedOrbit(null); setOrbitConstraints(null); }}>
                                  ← Back to Orbits
                                </button>

                                <div className="card mb-md bg-surface-hover">
                                  <div className="flex-between mb-sm">
                                    <span className="font-bold text-accent">Selected Bond: {selectedOrbit.representative.atom_i} → {selectedOrbit.representative.atom_j}</span>
                                    <span className="font-mono text-xs">[{selectedOrbit.representative.offset.join(',')}]</span>
                                  </div>
                                  {orbitConstraints && (
                                    <div className="grid gap-md">
                                      <div>
                                        <div className="text-xs uppercase font-bold opacity-60 mb-xs">Allowed Matrix Form</div>
                                        <div className="exchange-matrix-grid">
                                          {orbitConstraints.symbolic_matrix.flat().map((cell, c) => (
                                            <div key={c} className={`exchange-matrix-cell ${cell === '0' || cell === '0.0' ? 'zero' : ''}`}>
                                              {cell}
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                      <div>
                                        <div className="text-xs uppercase font-bold opacity-60 mb-xs">Free Parameters</div>
                                        <div className="flex gap-sm flex-wrap">
                                          {orbitConstraints.free_parameters.length > 0 ? orbitConstraints.free_parameters.map(p => (
                                            <span key={p} className="badge badge-primary">{p}</span>
                                          )) : <span className="text-xs italic opacity-50">None (Fixed by symmetry)</span>}
                                        </div>
                                      </div>

                                      {orbitConstraints.is_centrosymmetric && (
                                        <div className="alert alert-info py-xs px-sm text-xs">
                                          Bond has inversion symmetry (No DM allowed).
                                        </div>
                                      )}

                                      <button className="btn btn-primary w-full mt-sm" onClick={() => handleAddSymmetryInteraction(selectedOrbit, orbitConstraints)}>
                                        add Interaction Rule (Matrix)
                                      </button>
                                    </div>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Legacy Neighbor Suggestions (Optional keep or remove, keeping for fallback) */}
                      {neighborDistances && neighborDistances.length > 0 ? (
                        <div className="suggestion-grid">
                          {neighborDistances.map((n, i) => (
                            <div key={i} className="suggestion-card animate-fade-in" style={{ animationDelay: `${i * 0.05}s` }}>
                              <div className="suggestion-header">
                                <div className="distance-badge">{n.distance.toFixed(4)} Å</div>
                                <div className="pair-count">
                                  <Box size={14} />
                                  {n.count} Pairs
                                </div>
                              </div>
                              <div className="shell-info">
                                <span className="ref-pair-label">Reference Bond</span>
                                {n.equivalent_bonds && n.equivalent_bonds.length > 1 ? (
                                  <select
                                    className="minimal-select text-xs mt-xs"
                                    value={selectedBondIdxs[i] || 0}
                                    onChange={(e) => setSelectedBondIdxs({ ...selectedBondIdxs, [i]: parseInt(e.target.value) })}
                                  >
                                    {n.equivalent_bonds.map((b, bidx) => (
                                      <option key={bidx} value={bidx}>
                                        {b.pair.join(' → ')} [{b.offset.join(',')}]
                                      </option>
                                    ))}
                                  </select>
                                ) : (
                                  <span className="ref-pair-value">{n.ref_pair ? n.ref_pair.join(' → ') : 'N/A'}</span>
                                )}
                              </div>
                              <div className="suggestion-footer">
                                <button className="btn btn-primary btn-xs" onClick={() => {
                                  const bidx = selectedBondIdxs[i] || 0;
                                  const chosenBond = n.equivalent_bonds ? n.equivalent_bonds[bidx] : { pair: n.ref_pair, offset: n.offset };

                                  const nextRules = [...config.symmetry_interactions, {
                                    type: 'heisenberg',
                                    distance: Number(n.distance.toFixed(5)),
                                    value: `J${i + 1}`,
                                    ref_pair: chosenBond.pair,
                                    offset: chosenBond.offset
                                  }];
                                  const nextParams = { ...config.parameters };
                                  if (!nextParams[`J${i + 1}`]) nextParams[`J${i + 1}`] = 0.0;
                                  setConfig({ ...config, symmetry_interactions: nextRules, parameters: nextParams });
                                  showNotify(`Added Interaction Rule J${i + 1}`, 'success');
                                }}>
                                  <Plus size={12} className="mr-xs" />
                                  Add J{i + 1}
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="glass p-lg rounded-xl center">
                          <Info size={32} className="text-muted mb-md mx-auto opacity-20" />
                          <p className="text-sm text-muted italic">Click the button above to analyze the crystal structure and find neighbor shell distances.</p>
                        </div>
                      )}
                    </div>

                  </>
                ) : (
                  <>
                    <h2 className="section-title compact mb-xl">Explicit Interactions (Manual)</h2>

                    <div className="flex align-center gap-md mb-lg">
                      <div className="modern-toggle-group">
                        <button
                          className={`toggle-btn ${interactionMode === 'symmetry' ? 'active' : ''}`}
                          onClick={() => setInteractionMode('symmetry')}
                        >
                          <Wind size={14} className="mr-xs" />
                          Symmetry Rules
                        </button>
                        <button
                          className={`toggle-btn ${interactionMode === 'explicit' ? 'active' : ''}`}
                          onClick={() => setInteractionMode('explicit')}
                        >
                          <Activity size={14} className="mr-xs" />
                          Explicit Interactions
                        </button>
                      </div>

                      <button className="btn btn-primary btn-sm" onClick={() => {
                        const next = [...(config.explicit_interactions || [])];
                        next.push({ type: 'heisenberg', distance: 3.0, value: "J1", atom_i: 0, atom_j: 1, offset_j: [0, 0, 0] });
                        setConfig({ ...config, explicit_interactions: next });
                      }}><Plus size={14} /> Add Interaction</button>
                    </div>
                    <div className="interaction-grid">
                      {(config.explicit_interactions || []).map((inter, idx) => (
                        <div key={idx} className="interaction-card animate-fade-in">
                          <div className="interaction-header">
                            <div className="interaction-info">
                              <div className="interaction-icon-box">
                                {inter.type === 'heisenberg' ? <Zap size={16} /> : (inter.type.includes('anisotropic') ? <Crosshair size={16} /> : <Wind size={16} />)}
                              </div>
                              <div>
                                <span className="interaction-type">
                                  {inter.type === 'heisenberg' ? 'Heisenberg' : (inter.type.includes('anisotropic') ? 'Anisotropic' : 'DM Interaction')}
                                </span>
                                <span className="interaction-subtitle">Atoms: {inter.atom_i} → {inter.atom_j}</span>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <button
                                className={`icon-btn ${hiddenBondLabels.has(getBondKey(inter.value)) ? 'opacity-50' : 'text-primary'}`}
                                onClick={() => toggleBondVisibility(inter.value)}
                                title={hiddenBondLabels.has(getBondKey(inter.value)) ? "Show bonds" : "Hide bonds"}
                              >
                                {hiddenBondLabels.has(getBondKey(inter.value)) ? <EyeOff size={14} /> : <Eye size={14} />}
                              </button>
                              <button onClick={() => {
                                const next = config.explicit_interactions.filter((_, i) => i !== idx);
                                setConfig({ ...config, explicit_interactions: next });
                              }} className="icon-btn text-error"><Trash2 size={14} /></button>
                            </div>
                          </div>

                          <div className="interaction-params">
                            <div className="input-group">
                              <label>Type</label>
                              <select className="minimal-select" value={inter.type} onChange={(e) => {
                                const next = [...(config.explicit_interactions || [])];
                                next[idx].type = e.target.value;
                                if (e.target.value === 'heisenberg') {
                                  next[idx].value = "J1";
                                } else {
                                  // Both DM and Anisotropic use vector/array values
                                  next[idx].value = ["0", "0", "0"];
                                }
                                setConfig({ ...config, explicit_interactions: next });
                              }}>
                                <option value="heisenberg">Heisenberg</option>
                                <option value="dm_manual">DM Manual</option>
                                <option value="anisotropic_exchange">Anisotropic</option>
                              </select>
                            </div>
                            <div className="input-group">
                              <label>Distance</label>
                              <input type="number" step="0.01" className="minimal-input" value={typeof inter.distance === 'number' ? Number(inter.distance.toFixed(5)) : inter.distance} onChange={(e) => {
                                const next = [...config.explicit_interactions]; next[idx].distance = parseFloat(e.target.value); setConfig({ ...config, explicit_interactions: next })
                              }} />
                            </div>
                            <div className="input-group">
                              <label>Value / Vector</label>
                              {Array.isArray(inter.value) ? (
                                <div className="vector-input-grid">
                                  {(inter.type.includes('anisotropic') ? ['Jx', 'Jy', 'Jz'] : ['Dx', 'Dy', 'Dz']).map((label, k) => (
                                    <input
                                      key={k}
                                      type="text"
                                      className="table-input center"
                                      value={inter.value[k]}
                                      onChange={(e) => {
                                        const next = [...config.explicit_interactions]; next[idx].value[k] = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                      }}
                                      placeholder={label}
                                    />
                                  ))}
                                </div>
                              ) : (
                                <input type="text" className="minimal-input accent-text" value={inter.value} onChange={(e) => {
                                  const next = [...config.explicit_interactions]; next[idx].value = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                }} />
                              )}
                            </div>
                          </div>

                          <div className="interaction-footer">
                            <div className="flex-gap-sm align-center">
                              <span className="text-xxs opacity-60 font-bold uppercase">Offset J:</span>
                              {[0, 1, 2].map(k => (
                                <input key={k} type="number" className="table-input center w-10" value={inter.offset_j ? inter.offset_j[k] : 0} onChange={(e) => {
                                  const next = [...config.explicit_interactions];
                                  if (!next[idx].offset_j) next[idx].offset_j = [0, 0, 0];
                                  next[idx].offset_j[k] = parseInt(e.target.value);
                                  setConfig({ ...config, explicit_interactions: next })
                                }} />
                              ))}
                            </div>
                            <div className="flex-gap-sm align-center">
                              <span className="text-xxs opacity-60 font-bold uppercase">Indices:</span>
                              <input type="number" className="table-input center w-12" value={inter.atom_i} onChange={(e) => {
                                const next = [...config.explicit_interactions]; next[idx].atom_i = parseInt(e.target.value); setConfig({ ...config, explicit_interactions: next })
                              }} />
                              <ChevronRight size={10} className="opacity-40" />
                              <input type="number" className="table-input center w-12" value={inter.atom_j} onChange={(e) => {
                                const next = [...config.explicit_interactions]; next[idx].atom_j = parseInt(e.target.value); setConfig({ ...config, explicit_interactions: next })
                              }} />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'params' && (
              <div className="form-section">
                <h2 className="section-title mb-xl">Environment Settings</h2>
                <div className="card mb-lg">
                  <div className="grid-form">
                    <div className="input-group">
                      <label>Applied Field (T)</label>
                      <input type="number" value={config.parameters.H_mag} className="minimal-input" onChange={(e) => updateField('parameters', 'H_mag', parseFloat(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Field Direction (h,k,l)</label>
                      <div className="flex-gap-xs">
                        {[0, 1, 2].map(i => (
                          <input key={i} type="number" value={config.parameters.H_dir[i]} className="minimal-input" onChange={(e) => {
                            const next = [...config.parameters.H_dir]; next[i] = parseFloat(e.target.value); updateField('parameters', 'H_dir', next)
                          }} />
                        ))}
                      </div>
                    </div>

                  </div>
                </div>

                <div className="flex-between mb-md">
                  <h2 className="section-title">Model Parameters</h2>
                  {!isAddingParam ? (
                    <button className="btn btn-primary btn-sm" onClick={() => {
                      setIsAddingParam(true)
                      setNewParamName('')
                    }}><Plus size={14} /> Add Parameter</button>
                  ) : (
                    <div className="flex-gap-sm">
                      <input
                        type="text"
                        className="minimal-input"
                        placeholder="Name (e.g. J3)"
                        value={newParamName}
                        onChange={(e) => setNewParamName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && newParamName.trim()) {
                            setConfig(prev => ({
                              ...prev,
                              parameters: { ...prev.parameters, [newParamName.trim()]: 0.0 }
                            }))
                            setIsAddingParam(false)
                          }
                        }}
                        autoFocus
                        style={{ width: '120px' }}
                      />
                      <button className="icon-btn text-success" onClick={() => {
                        if (newParamName.trim()) {
                          setConfig(prev => ({
                            ...prev,
                            parameters: { ...prev.parameters, [newParamName.trim()]: 0.0 }
                          }))
                          setIsAddingParam(false)
                        }
                      }}><Check size={16} /></button>
                      <button className="icon-btn text-error" onClick={() => setIsAddingParam(false)}><Trash2 size={16} /></button>
                    </div>
                  )}
                </div>
                <div className="card">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Value</th>
                        <th style={{ width: '40px' }}></th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(config.parameters)
                        .filter(([key]) => !['S', 'H_mag', 'H_dir'].includes(key))
                        .map(([key, value]) => (
                          <tr key={key}>
                            <td className="mono">{key}</td>
                            <td>
                              <input
                                type="number"
                                step="0.001"
                                className="table-input"
                                value={value}
                                onChange={(e) => {
                                  setConfig(prev => ({
                                    ...prev,
                                    parameters: { ...prev.parameters, [key]: parseFloat(e.target.value) }
                                  }))
                                }}
                              />
                            </td>
                            <td>
                              <button
                                className="icon-btn text-error"
                                onClick={() => {
                                  const next = { ...config.parameters }
                                  delete next[key]
                                  setConfig(prev => ({ ...prev, parameters: next }))
                                }}
                              ><Trash2 size={14} /></button>
                            </td>
                          </tr>
                        ))}
                      {Object.keys(config.parameters).filter(k => !['S', 'H_mag', 'H_dir'].includes(k)).length === 0 && (
                        <tr>
                          <td colSpan="3" className="center text-secondary py-md">No model parameters defined.</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {activeTab === 'tasks' && (
              <div className="form-section">
                <h2 className="section-title">Tasks & Plotting</h2>
                <div className="grid-2 mt-md" style={{ alignItems: 'start' }}>
                  <div className="flex-col gap-lg">
                    <div className="card shadow-glow">
                      <h3>Calculation Tasks</h3>
                      <div className="task-cards-grid">
                        <div
                          className={`task-card ${config.tasks.minimization ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'minimization', !config.tasks.minimization)}
                        >
                          <div className="task-icon-box">
                            <Magnet size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Run Minimization</span>
                            <span className="task-desc">Calculate results</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.dispersion ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'dispersion', !config.tasks.dispersion)}
                        >
                          <div className="task-icon-box">
                            <Activity size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Dispersion</span>
                            <span className="task-desc">Calculate & Plot</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.sqw_map ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'sqw_map', !config.tasks.sqw_map)}
                        >
                          <div className="task-icon-box">
                            <BarChart2 size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">S(Q,ω) Map</span>
                            <span className="task-desc">Full spectral map</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.powder_average ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'powder_average', !config.tasks.powder_average)}
                        >
                          <div className="task-icon-box">
                            <Wind size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Powder Average</span>
                            <span className="task-desc">S(Q,ω) Sphere Sampl.</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.corrections ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'corrections', !config.tasks.corrections)}
                        >
                          <div className="task-icon-box">
                            <Target size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">1/S Corrections</span>
                            <span className="task-desc">Zero-point energy + moment reduction</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.scga ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'scga', !config.tasks.scga)}
                        >
                          <div className="task-icon-box">
                            <Search size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Diffuse S(q) — SCGA</span>
                            <span className="task-desc">Paramagnetic, above T_N</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.thermal_mc ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'thermal_mc', !config.tasks.thermal_mc)}
                        >
                          <div className="task-icon-box">
                            <Beaker size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Thermal Monte-Carlo</span>
                            <span className="task-desc">E, C, M, χ vs T (parallel tempering)</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.sampled_correlations ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'sampled_correlations', !config.tasks.sampled_correlations)}
                        >
                          <div className="task-icon-box">
                            <Play size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">Classical Dynamics S(q,ω)</span>
                            <span className="task-desc">Finite-T lineshapes (Landau–Lifshitz)</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>

                        <div
                          className={`task-card ${config.tasks.kpm_sqw ? 'active' : ''}`}
                          onClick={() => updateField('tasks', 'kpm_sqw', !config.tasks.kpm_sqw)}
                        >
                          <div className="task-icon-box">
                            <Zap size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">KPM S(q,ω)</span>
                            <span className="task-desc">No-diagonalization (SU(N)/entangled)</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>
                      </div>

                      {(config.tasks.scga || config.tasks.thermal_mc ||
                        config.tasks.sampled_correlations || config.tasks.kpm_sqw) && (
                        <div className="mt-md">
                          <h4 className="mb-sm">Beyond-LSWT settings</h4>
                          {config.tasks.scga && (
                            <div className="grid-form mb-md">
                              <div className="input-group">
                                <label>SCGA kT (meV)</label>
                                <input type="number" step="0.1" className="minimal-input"
                                  value={config.scga.temperature}
                                  onChange={(e) => updateField('scga', 'temperature', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>SCGA mesh density</label>
                                <input type="number" step="1" className="minimal-input"
                                  value={config.scga.mesh_density}
                                  onChange={(e) => updateField('scga', 'mesh_density', e.target.value)} />
                              </div>
                            </div>
                          )}
                          {config.tasks.thermal_mc && (
                            <div className="grid-form mb-md">
                              <div className="input-group">
                                <label>MC temperatures (meV, comma-sep)</label>
                                <input type="text" className="minimal-input"
                                  value={config.thermal_mc.temperatures}
                                  onChange={(e) => updateField('thermal_mc', 'temperatures', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>MC supercell (L1,L2,L3)</label>
                                <input type="text" className="minimal-input"
                                  value={config.thermal_mc.supercell}
                                  onChange={(e) => updateField('thermal_mc', 'supercell', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>MC sweeps</label>
                                <input type="number" step="500" className="minimal-input"
                                  value={config.thermal_mc.n_sweeps}
                                  onChange={(e) => updateField('thermal_mc', 'n_sweeps', e.target.value)} />
                              </div>
                            </div>
                          )}
                          {config.tasks.sampled_correlations && (
                            <div className="grid-form mb-md">
                              <div className="input-group">
                                <label>Dynamics kT (meV)</label>
                                <input type="number" step="0.1" className="minimal-input"
                                  value={config.sampled_correlations.temperature}
                                  onChange={(e) => updateField('sampled_correlations', 'temperature', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>Dynamics supercell (L1,L2,L3)</label>
                                <input type="text" className="minimal-input"
                                  value={config.sampled_correlations.supercell}
                                  onChange={(e) => updateField('sampled_correlations', 'supercell', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>Trajectories</label>
                                <input type="number" step="1" className="minimal-input"
                                  value={config.sampled_correlations.n_traj}
                                  onChange={(e) => updateField('sampled_correlations', 'n_traj', e.target.value)} />
                              </div>
                            </div>
                          )}
                          {config.tasks.kpm_sqw && (
                            <div className="grid-form mb-md">
                              <div className="input-group">
                                <label>KPM E max (meV)</label>
                                <input type="number" step="1" className="minimal-input"
                                  value={config.kpm.e_max}
                                  onChange={(e) => updateField('kpm', 'e_max', e.target.value)} />
                              </div>
                              <div className="input-group">
                                <label>KPM FWHM (meV)</label>
                                <input type="number" step="0.05" className="minimal-input"
                                  value={config.kpm.fwhm}
                                  onChange={(e) => updateField('kpm', 'fwhm', e.target.value)} />
                              </div>
                            </div>
                          )}
                          <p className="text-xs opacity-50">
                            SCGA / thermal MC / dynamics are classical (paramagnetic-friendly):
                            run alone they skip the LSWT ground-state check. KPM needs the
                            SU(N) or entangled engine.
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="card shadow-glow">
                      <h3>Display Parameters</h3>
                      <div className="grid-form mt-md">
                        <div className="input-group">
                          <label>Energy Min <span style={{ textTransform: 'none' }}>(meV)</span></label>
                          <input type="number" step="0.1" value={config.plotting.energy_min} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'energy_min', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Energy Max <span style={{ textTransform: 'none' }}>(meV)</span></label>
                          <input type="number" step="0.1" value={config.plotting.energy_max} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'energy_max', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Broadening <span style={{ textTransform: 'none' }}>(meV)</span></label>
                          <input type="number" step="0.01" value={config.plotting.broadening} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'broadening', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Energy Res. <span style={{ textTransform: 'none' }}>(meV)</span></label>
                          <input type="number" step="0.01" value={config.plotting.energy_resolution} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'energy_resolution', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group" style={{ gridColumn: '1 / -1' }}>
                          <label>Momentum Max (Å⁻¹)</label>
                          <input type="number" step="0.1" value={config.plotting.momentum_max} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'momentum_max', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group" style={{ gridColumn: '1 / -1' }}>
                          <label>Visualization Targets</label>
                          <div className="flex-col gap-sm mt-xs">
                            <div
                              className={`task-card ${config.plotting.auto_scale_disp !== false ? 'active' : ''}`}
                              onClick={() => updateField('plotting', 'auto_scale_disp', !(config.plotting.auto_scale_disp !== false))}
                            >
                              <div className="task-icon-box">
                                <BarChart2 size={18} />
                              </div>
                              <div className="task-info">
                                <span className="task-name">Auto-scale Y-Axis</span>
                                <span className="task-desc">Dispersion fits energy range (ignores Energy Min/Max)</span>
                              </div>
                              <div className="task-check">
                                <Check size={12} strokeWidth={4} />
                              </div>
                            </div>

                            <div
                              className={`task-card ${config.plotting.show_plot !== false ? 'active' : ''}`}
                              onClick={() => updateField('plotting', 'show_plot', !(config.plotting.show_plot !== false))}
                            >
                              <div className="task-icon-box">
                                <Eye size={18} />
                              </div>
                              <div className="task-info">
                                <span className="task-name">Show Plot</span>
                                <span className="task-desc">Energy dispersion / Sq(w)</span>
                              </div>
                              <div className="task-check">
                                <Check size={12} strokeWidth={4} />
                              </div>
                            </div>

                            <div
                              className={`task-card ${config.plotting.plot_structure || false ? 'active' : ''}`}
                              onClick={() => updateField('plotting', 'plot_structure', !config.plotting.plot_structure)}
                            >
                              <div className="task-icon-box">
                                <Box size={18} />
                              </div>
                              <div className="task-info">
                                <span className="task-name">Show Structure</span>
                                <span className="task-desc">3D Crystal View</span>
                              </div>
                              <div className="task-check">
                                <Check size={12} strokeWidth={4} />
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex-col gap-lg">
                    <div className="card shadow-glow">
                      <h3>Minimization Parameters</h3>
                      <div className="grid-form mt-md">
                        <div className="input-group">
                          <label>Method</label>
                          <select
                            className="minimal-select"
                            value={config.minimization.method}
                            onChange={(e) => onMinimizationMethodChange(e.target.value)}
                          >
                            <option value="anneal">Monte-Carlo annealing (recommended)</option>
                            <option value="steep">Steepest descent (local field)</option>
                            <option value="L-BFGS-B">L-BFGS-B (gradient multistart)</option>
                            <option value="TNC">TNC (gradient multistart)</option>
                            <option value="SLSQP">SLSQP (gradient multistart)</option>
                          </select>
                          <div className="text-xs text-secondary mt-1" style={{ gridColumn: "1 / -1" }}>
                            {isAnnealMethod(config.minimization.method)
                              ? (config.minimization.method === 'steep'
                                  ? 'Aligns each spin with its local field (SpinW optmagsteep). Fast, but it only goes downhill \u2014 it cannot escape a local minimum.'
                                  : 'Metropolis + cooling (SpinW anneal / Sunny LocalSampler), then a gradient polish. Crosses barriers, so it does not get trapped.')
                              : 'Random multistart in (\u03b8, \u03c6). Gets trapped on frustrated systems \u2014 prefer annealing.'}
                          </div>
                        </div>

                        <div className="input-group">
                          <label>{isAnnealMethod(config.minimization.method) ? 'Runs' : 'Num Starts'}</label>
                          <input type="number" value={config.minimization.num_starts} className="minimal-input"
                            onChange={(e) => updateField('minimization', 'num_starts', parseInt(e.target.value))} />
                        </div>

                        {config.minimization.method === 'anneal' && (
                          <div className="input-group">
                            <label>Sweeps</label>
                            <input type="number" value={config.minimization.n_sweeps ?? 2000} className="minimal-input"
                              onChange={(e) => updateField('minimization', 'n_sweeps', parseInt(e.target.value))} />
                            <div className="text-xs text-secondary mt-1" style={{ gridColumn: "1 / -1" }}>
                              Cooling steps; each attempts one move per spin.
                            </div>
                          </div>
                        )}

                        {!isAnnealMethod(config.minimization.method) && (
                          <>
                            <div className="input-group">
                              <label>N Workers</label>
                              <input type="number" value={config.minimization.n_workers} className="minimal-input"
                                onChange={(e) => updateField('minimization', 'n_workers', parseInt(e.target.value))} />
                            </div>
                            <div className="input-group">
                              <label>Early Stopping</label>
                              <input type="number" value={config.minimization.early_stopping} className="minimal-input"
                                onChange={(e) => updateField('minimization', 'early_stopping', parseInt(e.target.value))} />
                              <div className="text-xs text-warning mt-1" style={{ gridColumn: "1 / -1" }}>
                                Stop after N starts hit the same minimum. Use &ge; 2 &times; the number of magnetic sites; too low silently returns a LOCAL minimum.
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {config.tasks.powder_average && config.powder_average && (
                      <div className="card shadow-glow animate-slide-in">
                        <h3>Powder Average Settings</h3>
                        <div className="grid-form mt-md">
                          <div className="input-group">
                            <label>Q Min (Å⁻¹)</label>
                            <input type="number" step="0.1" value={config.powder_average.q_min} className="minimal-input"
                              onChange={(e) => updateField('powder_average', 'q_min', parseFloat(e.target.value))} />
                          </div>
                          <div className="input-group">
                            <label>Q Max (Å⁻¹)</label>
                            <input type="number" step="0.1" value={config.powder_average.q_max} className="minimal-input"
                              onChange={(e) => updateField('powder_average', 'q_max', parseFloat(e.target.value))} />
                          </div>
                          <div className="input-group">
                            <label>Q Points</label>
                            <input type="number" value={config.powder_average.q_count} className="minimal-input"
                              onChange={(e) => updateField('powder_average', 'q_count', parseInt(e.target.value))} />
                          </div>
                          <div className="input-group">
                            <label>Num Samples</label>
                            <input type="number" value={config.powder_average.num_samples} className="minimal-input"
                              onChange={(e) => updateField('powder_average', 'num_samples', parseInt(e.target.value))} />
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="card shadow-glow">
                      <h3>Calculation Settings</h3>
                      <div className="grid-form mt-md">
                        <div className="input-group">
                          <label>Cache Mode</label>
                          <select
                            value={config.calculation.cache_mode}
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'cache_mode', e.target.value)}
                          >
                            <option value="none">None (No Caching)</option>
                            <option value="auto">Auto (Smart Caching)</option>
                            <option value="r">Read (Force Read Cache)</option>
                            <option value="w">Write (Force Regeneration)</option>
                          </select>
                          <p className="text-xs opacity-50 mt-xs">
                            'None' is recommended for small systems or when debugging.
                          </p>
                        </div>
                        <div className="input-group">
                          <label>Compute Backend</label>
                          <select
                            value={config.calculation.backend || 'numpy'}
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'backend', e.target.value)}
                          >
                            <option value="numpy">NumPy (default)</option>
                            <option value="fortran">Fortran (fMagCalc)</option>
                          </select>
                          <p className="text-xs opacity-50 mt-xs">
                            'Fortran' uses the fMagCalc backend for S(Q,ω) and powder
                            (much faster); falls back to NumPy if it isn't installed.
                          </p>
                        </div>

                        <div className="input-group">
                          <label>Ground-State Check</label>
                          <select
                            value={config.calculation.on_imaginary || 'error'}
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'on_imaginary', e.target.value)}
                          >
                            <option value="error">Fail the run (default)</option>
                            <option value="warn">Warn only (metastable structure)</option>
                            <option value="off">Disable</option>
                          </select>
                          <p className="text-xs opacity-50 mt-xs">
                            Spin waves are an expansion about a classical energy <em>minimum</em>;
                            about anything else the spectrum is meaningless. Three guards run:
                            imaginary magnon energies, a lower-energy relaxation, and a q≠0 spiral
                            instability (a longer-period ground state the cell can't hold). The run
                            fails if any fires.
                          </p>
                          {config.calculation.on_imaginary === 'warn' && (
                            <p className="text-xs text-warning mt-xs">
                              Use only when the structure is <strong>knowingly</strong> metastable —
                              e.g. a commensurate approximation to an incommensurate spiral, or a
                              state the reference calculation also treats as non-minimal. Otherwise
                              you are silently computing the wrong physics.
                            </p>
                          )}
                          {config.calculation.on_imaginary === 'off' && (
                            <p className="text-xs text-warning mt-xs">
                              All three ground-state guards are disabled. A wrong ground state will
                              now produce a plausible-looking but meaningless spectrum, with no warning.
                            </p>
                          )}
                        </div>

                        <div className="input-group">
                          <label>LSWT Engine</label>
                          <select
                            value={config.calculation.mode || 'dipole'}
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'mode', e.target.value)}
                          >
                            <option value="dipole">Dipole (default)</option>
                            <option value="SUN">SU(N) — single-ion / multipolar</option>
                            <option value="entangled">Entangled units — dimers / trimers</option>
                          </select>
                          <p className="text-xs opacity-50 mt-xs">
                            SU(N) captures single-ion (multipolar) excitations — e.g. FeI₂'s
                            bound state — that dipole LSWT cannot represent. Use it for S ≥ 1
                            with strong single-ion anisotropy.
                          </p>
                          {config.calculation.mode === 'SUN' && (
                            <p className="text-xs text-warning mt-xs">
                              SU(N)'s ground state differs from the dipole one — enable
                              <strong> Run Minimization</strong> so it is found in SU(N), not
                              inherited. Powder/domain averaging are not yet supported.
                            </p>
                          )}
                          {config.calculation.mode === 'entangled' && (
                            <div className="mt-xs">
                              <div className="input-group">
                                <label>Units (JSON, site indices per unit)</label>
                                <input type="text" className="minimal-input"
                                  placeholder='e.g. [[0,1],[2,3]] — blank: from config'
                                  value={config.calculation.units_text || ''}
                                  onChange={(e) => updateField('calculation', 'units_text', e.target.value)} />
                              </div>
                              <div className="input-group mt-xs">
                                <label>Dimer series order (0 = harmonic)</label>
                                <input type="number" step="1" min="0" className="minimal-input"
                                  value={config.calculation.series_order || 0}
                                  onChange={(e) => updateField('calculation', 'series_order',
                                    parseInt(e.target.value) || 0)} />
                              </div>
                              <p className="text-xs opacity-50 mt-xs">
                                Each unit (e.g. a singlet dimer) becomes one effective SU(N)
                                site; excitations are its triplons. Orders ≥ 4 of the series
                                get expensive — see TUTORIAL §4g.
                              </p>
                            </div>
                          )}
                        </div>

                        <div className="input-group">
                          <label>Temperature (K)</label>
                          <input
                            type="number" step="1" min="0"
                            value={config.calculation.temperature ?? ''}
                            placeholder="0 (T → 0)"
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'temperature',
                              e.target.value === '' ? null : parseFloat(e.target.value))}
                          />
                          <p className="text-xs opacity-50 mt-xs">
                            Applies the Bose thermal factor to S(Q,ω)/powder intensities.
                            Blank = T → 0 (bare LSWT).
                          </p>
                        </div>

                        <div className="input-group">
                          <label>Cross-section</label>
                          <select
                            value={config.calculation.cross_section || 'perp'}
                            className="minimal-input"
                            onChange={(e) => updateField('calculation', 'cross_section', e.target.value)}
                          >
                            <option value="perp">Unpolarized ⊥ (default)</option>
                            <option value="trace">Trace (full)</option>
                            <option value="chiral">Chiral</option>
                            <option value="xx">Sˣˣ</option>
                            <option value="yy">Sʸʸ</option>
                            <option value="zz">Sᶻᶻ</option>
                          </select>
                          <p className="text-xs opacity-50 mt-xs">
                            Neutron cross-section contraction for S(Q,ω) intensities.
                          </p>
                        </div>
                      </div>

                      <h3 className="mt-lg">Data Export</h3>
                      <div className="mt-md">
                        <label className="flex-between align-center glass rounded-lg border-light mb-md modern-switch-container pointer" style={{ padding: '8px 12px', display: 'flex' }}>
                          <div className="flex align-center gap-md">
                            <Download size={18} className="text-accent" />
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                              <span className="block font-bold">Export Numeric Results (.npz)</span>
                              <span className="text-xxs opacity-60 mt-xs" style={{ fontSize: '10px' }}>Save raw eigenvalues and intensities to binary files.</span>
                            </div>
                          </div>
                          <label className="modern-switch" style={{ marginBottom: 0 }}>
                            <input
                              type="checkbox"
                              checked={config.output.save_data}
                              onChange={(e) => updateField('output', 'save_data', e.target.checked)}
                            />
                            <span className="switch-slider"></span>
                          </label>
                        </label>

                        {config.output.save_data && (
                          <div className="grid-form animate-fade-in mb-md">
                            <div className="input-group">
                              <label>Dispersion NPZ</label>
                              <input
                                type="text"
                                value={config.output.disp_data_filename}
                                className="minimal-input"
                                placeholder="disp_data.npz"
                                onChange={(e) => updateField('output', 'disp_data_filename', e.target.value)}
                              />
                            </div>
                            <div className="input-group">
                              <label>S(Q,w) NPZ</label>
                              <input
                                type="text"
                                value={config.output.sqw_data_filename}
                                className="minimal-input"
                                placeholder="sqw_data.npz"
                                onChange={(e) => updateField('output', 'sqw_data_filename', e.target.value)}
                              />
                            </div>
                          </div>
                        )}

                        <label className="flex-between align-center glass rounded-lg border-light mb-md modern-switch-container pointer" style={{ padding: '12px 16px', display: 'flex' }}>
                          <div className="flex align-center gap-md">
                            <Image size={18} className="text-accent" />
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                              <span className="block font-bold">Export Visual Plots (.png)</span>
                              <span className="text-xxs opacity-60 mt-xs" style={{ fontSize: '10px' }}>Save dispersion and S(Q,w) maps as image files.</span>
                            </div>
                          </div>
                          <label className="modern-switch" style={{ marginBottom: 0 }}>
                            <input
                              type="checkbox"
                              checked={config.plotting.save_plot}
                              onChange={(e) => updateField('plotting', 'save_plot', e.target.checked)}
                            />
                            <span className="switch-slider"></span>
                          </label>
                        </label>

                        {config.plotting.save_plot && (
                          <div className="grid-form animate-fade-in mb-md">
                            <div className="input-group">
                              <label>Dispersion Plot</label>
                              <input
                                type="text"
                                value={config.plotting.disp_plot_filename}
                                className="minimal-input"
                                placeholder="disp_plot.png"
                                onChange={(e) => updateField('plotting', 'disp_plot_filename', e.target.value)}
                              />
                            </div>
                            <div className="input-group">
                              <label>S(Q,w) Plot</label>
                              <input
                                type="text"
                                value={config.plotting.sqw_plot_filename}
                                className="minimal-input"
                                placeholder="sqw_plot.png"
                                onChange={(e) => updateField('plotting', 'sqw_plot_filename', e.target.value)}
                              />
                            </div>
                          </div>
                        )}

                        <label className="flex-between align-center glass rounded-lg border-light mb-md modern-switch-container pointer" style={{ padding: '12px 16px', display: 'flex' }}>
                          <div className="flex align-center gap-md">
                            <FileText size={18} className="vibrant-text" />
                            <span className="text-sm font-bold">Export results to CSV</span>
                          </div>
                          <label className="modern-switch" style={{ marginBottom: 0 }}>
                            <input
                              type="checkbox"
                              checked={config.tasks.export_csv}
                              onChange={(e) => updateField('tasks', 'export_csv', e.target.checked)}
                            />
                            <span className="switch-slider"></span>
                          </label>
                        </label>

                        {config.tasks.export_csv && (
                          <div className="grid-form animate-fade-in">
                            <div className="input-group">
                              <label>Dispersion CSV</label>
                              <input
                                type="text"
                                value={config.output.disp_csv_filename}
                                className="minimal-input"
                                onChange={(e) => updateField('output', 'disp_csv_filename', e.target.value)}
                              />
                            </div>
                            <div className="input-group">
                              <label>S(Q,w) CSV</label>
                              <input
                                type="text"
                                value={config.output.sqw_csv_filename}
                                className="minimal-input"
                                onChange={(e) => updateField('output', 'sqw_csv_filename', e.target.value)}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card mt-lg">
                  <div className="flex-between mb-md">
                    <h3>High Symmetry Points</h3>
                    <button className="btn btn-secondary btn-sm" onClick={() => {
                      const name = prompt("Enter point name (e.g. L):");
                      if (name) {
                        setConfig(prev => ({
                          ...prev,
                          q_path: {
                            ...prev.q_path,
                            points: { ...prev.q_path.points, [name]: [0, 0, 0] }
                          }
                        }));
                      }
                    }}><Plus size={14} /> Add Point</button>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Label</th>
                        <th>Coordinates (H, K, L)</th>
                        <th style={{ width: '40px' }}></th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(config.q_path.points).map(([label, pos], idx) => (
                        <tr key={label}>
                          <td className="mono">{label}</td>
                          <td>
                            <div className="flex-gap-xs">
                              {[0, 1, 2].map(i => (
                                <input
                                  key={i}
                                  type="number"
                                  step="0.001"
                                  value={pos[i]}
                                  className="table-input"
                                  onChange={(e) => {
                                    const nextPoints = { ...config.q_path.points };
                                    nextPoints[label][i] = parseFloat(e.target.value);
                                    setConfig({ ...config, q_path: { ...config.q_path, points: nextPoints } });
                                  }}
                                />
                              ))}
                            </div>
                          </td>
                          <td>
                            <button className="icon-btn text-error" onClick={() => {
                              const nextPoints = { ...config.q_path.points };
                              delete nextPoints[label];
                              setConfig({ ...config, q_path: { ...config.q_path, points: nextPoints } });
                            }}><Trash2 size={14} /></button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card mt-xl">
                  <h3>Q-Path Sequence</h3>
                  <div className="mt-md">
                    <div className="q-path-flow mb-xl">
                      {config.q_path.path.length === 0 && (
                        <span className="text-sm opacity-40 italic">Add points below to build your calculation path...</span>
                      )}
                      {config.q_path.path.map((p, idx) => (
                        <React.Fragment key={idx}>
                          <div className="q-path-node">
                            <span className="q-path-step-num">{idx + 1}</span>
                            {p}
                            <button className="icon-btn ml-xs" onClick={() => {
                              const nextPath = config.q_path.path.filter((_, i) => i !== idx);
                              setConfig({ ...config, q_path: { ...config.q_path, path: nextPath } });
                            }}><Trash2 size={12} /></button>
                          </div>
                          {idx < config.q_path.path.length - 1 && (
                            <div className="q-path-connector">
                              <ChevronRight size={16} />
                            </div>
                          )}
                        </React.Fragment>
                      ))}
                    </div>

                    <div className="flex-between align-center p-md glass rounded-lg border-light">
                      <div className="flex-gap-sm align-center">
                        <select className="minimal-select" id="point-select" style={{ width: '180px' }}>
                          <option value="">Select Point...</option>
                          {Object.keys(config.q_path.points).map(p => (
                            <option key={p} value={p}>{p}</option>
                          ))}
                        </select>
                        <button className="btn btn-primary btn-sm" onClick={() => {
                          const sel = document.getElementById('point-select');
                          if (sel.value) {
                            setConfig({ ...config, q_path: { ...config.q_path, path: [...config.q_path.path, sel.value] } });
                            sel.value = "";
                          }
                        }}><Plus size={14} /> Add to Path</button>
                      </div>

                      <div className="flex-gap-sm align-center">
                        <span className="text-xxs opacity-60 font-bold uppercase tracking-wider">Points per segment:</span>
                        <input
                          type="number"
                          value={config.q_path.points_per_segment}
                          onChange={(e) => updateField('q_path', 'points_per_segment', parseInt(e.target.value))}
                          className="minimal-input"
                          style={{ width: '80px', textAlign: 'center' }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {activeTab === 'magstruct' && (
              <div className="form-section">
                <div className="flex-between mb-md">
                  <h2 className="section-title">Magnetic Structure</h2>
                  <label className="flex-gap-sm align-center modern-switch-container pointer">
                    <span className="text-sm font-bold vibrant-text">Include Manual Structure</span>
                    <label className="modern-switch">
                      <input
                        type="checkbox"
                        checked={config.magnetic_structure.enabled}
                        onChange={(e) => updateField('magnetic_structure', 'enabled', e.target.checked)}
                      />
                      <span className="switch-slider"></span>
                    </label>
                  </label>
                </div>

                {!config.magnetic_structure.enabled && (
                  <div className="card glass opacity-60 text-center py-xl border-dashed">
                    <Magnet className="mx-auto mb-sm opacity-40" size={32} />
                    <p>Manual magnetic structure is currently disabled.</p>
                    <p className="text-xs mt-xs">Use the toggle above to enable manual spin direction input. If disabled, the calculation will rely on the optimizer to find the ground state.</p>
                  </div>
                )}

                {config.magnetic_structure.enabled && (
                  <div className="card mb-lg animate-fade-in">
                    <div className="input-group">
                      <label>Structure Type</label>
                      <select
                        className="minimal-select"
                        value={config.magnetic_structure.type}
                        onChange={(e) => updateField('magnetic_structure', 'type', e.target.value)}
                      >
                        <option value="pattern">Pattern Based</option>
                      </select>
                    </div>
                    {config.magnetic_structure.type === 'pattern' && (
                      <div className="mt-md">
                        <div className="input-group">
                          <label>Pattern Type</label>
                          <select
                            className="minimal-select"
                            value={config.magnetic_structure.pattern_type}
                            onChange={(e) => updateField('magnetic_structure', 'pattern_type', e.target.value)}
                          >
                            <option value="antiferromagnetic">Antiferromagnetic</option>
                            <option value="generic">Generic/Custom List</option>
                          </select>
                        </div>

                        <div className="mt-xl">
                          <div className="flex-between mb-md">
                            <h3>Spin Directions (Unit Vectors)</h3>
                            <button className="btn btn-secondary btn-sm" onClick={() => {
                              const next = [...config.magnetic_structure.directions];
                              next.push([1, 0, 0]);
                              updateField('magnetic_structure', 'directions', next);
                            }}><Plus size={14} /> Add Direction</button>
                          </div>
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th style={{ width: '60px' }}>#</th>
                                <th>Direction (Sx, Sy, Sz)</th>
                                <th style={{ width: '40px' }}></th>
                              </tr>
                            </thead>
                            <tbody>
                              {config.magnetic_structure.directions.map((dir, idx) => (
                                <tr key={idx}>
                                  <td className="center opacity-40">{idx}</td>
                                  <td>
                                    <div className="flex-gap-xs">
                                      {[0, 1, 2].map(i => (
                                        <input key={i} type="number" step="0.001" value={dir[i]} className="table-input" onChange={(e) => {
                                          const next = [...config.magnetic_structure.directions];
                                          next[idx][i] = parseFloat(e.target.value);
                                          updateField('magnetic_structure', 'directions', next);
                                        }} />
                                      ))}
                                    </div>
                                  </td>
                                  <td>
                                    <button className="icon-btn text-error" onClick={() => {
                                      const next = config.magnetic_structure.directions.filter((_, i) => i !== idx);
                                      updateField('magnetic_structure', 'directions', next);
                                    }}><Trash2 size={14} /></button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            {activeTab === 'fitting' && (
              <div className="form-section">
                <div className="flex-between mb-md">
                  <h2 className="section-title">Data Fitting</h2>
                  <button className="btn btn-primary btn-sm" onClick={runFit} disabled={calcLoading}>
                    <Target size={14} /> {calcLoading ? 'Fitting…' : 'Run Fit'}
                  </button>
                </div>

                <p className="text-xs opacity-70 mb-lg">
                  Fit the spin Hamiltonian to inelastic-neutron-scattering data. The
                  fit reuses one calculator and updates parameters each iteration.
                  Best-fit values, an lmfit report, and a data-vs-model comparison
                  plot appear in the Run &amp; Analyze tab.
                </p>

                <div className="card mb-lg">
                  <div className="input-group">
                    <label>Data Type</label>
                    <select className="minimal-select" value={config.fitting.type}
                      onChange={(e) => updateFitting({ type: e.target.value })}>
                      <option value="dispersion">Dispersion E(Q) — peak positions</option>
                      <option value="sqw">Single-crystal S(Q,ω) — intensities</option>
                      <option value="powder">Powder S(|Q|,ω) — intensities</option>
                    </select>
                  </div>

                  <div className="input-group mt-md">
                    <label>Experimental Data File</label>
                    <div className="flex-gap-sm align-center">
                      <input type="file" accept=".txt,.csv,.dat,.npz"
                        onChange={handleFitDataUpload} style={{ flex: 1 }} />
                      {config.fitting.data_label && (
                        <span className="text-xs vibrant-text">{config.fitting.data_label}</span>
                      )}
                    </div>
                    <p className="text-xs opacity-60 mt-xs">
                      {config.fitting.type === 'dispersion'
                        ? 'Columns: h, k, l, E, sigma [, mode]  (comma-separated, # comments)'
                        : config.fitting.type === 'sqw'
                          ? 'Columns: h, k, l, energy, intensity, error'
                          : 'Columns: |Q|, energy, intensity, error'}
                    </p>
                  </div>

                  <div className="flex-gap-sm mt-md">
                    <div className="input-group" style={{ flex: 1 }}>
                      <label>Method</label>
                      <select className="minimal-select" value={config.fitting.method}
                        onChange={(e) => updateFitting({ method: e.target.value })}>
                        <option value="leastsq">Levenberg–Marquardt (leastsq)</option>
                        <option value="least_squares">Trust Region (least_squares)</option>
                        <option value="nelder">Nelder–Mead</option>
                        <option value="differential_evolution">Differential Evolution</option>
                      </select>
                    </div>
                    {config.fitting.type === 'dispersion' && (
                      <div className="input-group" style={{ flex: 1 }}>
                        <label>Band Assignment</label>
                        <select className="minimal-select" value={config.fitting.match}
                          onChange={(e) => updateFitting({ match: e.target.value })}>
                          <option value="nearest">Nearest band (no mode column)</option>
                          <option value="mode">Use mode column</option>
                        </select>
                      </div>
                    )}
                  </div>
                </div>

                <div className="card mb-lg">
                  <h3 className="mb-sm">Parameters to Fit</h3>
                  {fittableParams.length === 0 ? (
                    <p className="text-xs opacity-60">No scalar parameters defined.</p>
                  ) : (
                    <table className="data-table">
                      <thead>
                        <tr><th>Vary</th><th>Name</th><th>Start</th><th>Min</th><th>Max</th></tr>
                      </thead>
                      <tbody>
                        {fittableParams.map((name) => {
                          const varied = (config.fitting.vary || []).includes(name)
                          const b = (config.fitting.bounds || {})[name] || [null, null]
                          return (
                            <tr key={name}>
                              <td>
                                <input type="checkbox" checked={varied}
                                  onChange={() => toggleVaryParam(name)} />
                              </td>
                              <td className="font-bold">{name}</td>
                              <td>{config.parameters[name]}</td>
                              <td>
                                <input type="number" className="minimal-input" style={{ width: '70px' }}
                                  disabled={!varied} value={b[0] ?? ''}
                                  onChange={(e) => setParamBound(name, 0, e.target.value)} />
                              </td>
                              <td>
                                <input type="number" className="minimal-input" style={{ width: '70px' }}
                                  disabled={!varied} value={b[1] ?? ''}
                                  onChange={(e) => setParamBound(name, 1, e.target.value)} />
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  )}
                </div>

                {(config.fitting.type === 'sqw' || config.fitting.type === 'powder') && (
                  <div className="card mb-lg">
                    <h3 className="mb-sm">Intensity Model</h3>
                    {['scale', 'background', 'energy_broadening'].map((key) => (
                      <div className="flex-gap-sm align-center mb-sm" key={key}>
                        <span style={{ width: '150px' }} className="text-sm font-bold">
                          {key === 'energy_broadening' ? 'Energy broadening (FWHM)' : key}
                        </span>
                        <input type="number" className="minimal-input" style={{ width: '90px' }}
                          value={config.fitting[key].value}
                          onChange={(e) => updateFitting({ [key]: { ...config.fitting[key], value: parseFloat(e.target.value) } })} />
                        <label className="flex-gap-sm align-center pointer text-xs">
                          <input type="checkbox" checked={config.fitting[key].vary}
                            onChange={(e) => updateFitting({ [key]: { ...config.fitting[key], vary: e.target.checked } })} />
                          vary
                        </label>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {activeTab === 'run' && (
          <div className="content-area animate-fade-in full-width-tab">
            <div className="flex-between align-center mb-xl">
              <div>
                <h2 className="section-title mb-xs">Run Calculation & Analysis</h2>
                <p className="text-sm opacity-60">Execute the simulation using current settings and visualize results.</p>
              </div>
              <div className="flex align-center gap-md">
                {calcLoading ? (
                  <>
                    <span className="btn btn-secondary btn-lg shadow-glow opacity-80 pointer-events-none">
                      <Activity className="animate-spin mr-sm" /> Calculating...
                    </span>
                    <button
                      className={`btn btn-danger btn-lg shadow-glow ${calcStopping ? 'opacity-50 pointer-events-none' : ''}`}
                      onClick={stopCalculation}
                      disabled={calcStopping}
                    >
                      <Square size={16} className="mr-sm" /> {calcStopping ? 'Stopping...' : 'Stop'}
                    </button>
                  </>
                ) : (
                  <button
                    className="btn btn-primary btn-lg shadow-glow"
                    onClick={runCalculation}
                  >
                    <Play size={18} className="mr-sm" /> Run Calculation
                  </button>
                )}
              </div>
            </div>

            {calcError && (
              <div className="card border-error mb-xl bg-error-dim">
                <div className="flex align-center gap-md text-error">
                  <Info />
                  <div>
                    <h4 className="font-bold">Execution Failed</h4>
                    <p className="text-sm opacity-80 mt-xs font-mono">{calcError}</p>
                  </div>
                </div>
              </div>
            )}

            {calcResults && (
              <div className="animate-slide-up">
                <div className="card mb-xl border-success bg-success-dim">
                  <div className="flex align-center gap-md text-success">
                    <Check />
                    <span className="font-bold">Calculation Completed Successfully</span>
                  </div>
                </div>

                <div className="flex-col gap-xl">
                  {calcResults.plots.map((plotUrl, idx) => {
                    const isJson = plotUrl.endsWith('.json');
                    const isMagStructure = plotUrl.includes('mag_structure');

                    if (isJson && isMagStructure) {
                      if (!jsonCache[plotUrl]) {
                        loadStructureData(plotUrl);
                        return (
                          <div key={idx} className="card p-0 overflow-hidden shadow-lg">
                            <div className="p-xl text-center"><Activity className="animate-spin inline-block" /> Loading 3D View...</div>
                          </div>
                        );
                      }

                      return (
                        <div key={idx} className="card p-0 overflow-hidden shadow-lg">
                          <div className="p-sm glass border-b border-light flex-between">
                            <span className="font-bold text-sm uppercase tracking-wider opacity-70">Interactive Magnetic Structure</span>
                            <div className="flex-gap-sm align-center">
                              <button
                                className="btn btn-secondary btn-sm"
                                title="Save this minimized structure into the Manual Structure tab for reuse (disables minimization)"
                                onClick={() => importMinimizedStructure(jsonCache[plotUrl])}
                              >
                                <Wind size={14} /> Use as Manual Structure
                              </button>
                              <a href={plotUrl} download className="icon-btn" title="Download Data">
                                <Download size={14} />
                              </a>
                            </div>
                          </div>
                          <div className="plot-container bg-white">
                            <MagneticStructureViewer data={jsonCache[plotUrl]} />
                          </div>
                        </div>
                      );
                    }

                    // Skip static image if we have the JSON version and decide to only show one
                    // For now, let's render everything else normally. 
                    // If we have both PNG and JSON for mag structure, the server sends both.
                    // We might want to HIDE the PNG if we successfully rendered the JSON.
                    // But determining that across map iterations is hard. 
                    // Let's just hide the PNG version if it's "mag_structure.png" explicitly?
                    if (plotUrl.endsWith('mag_structure.png')) {
                      // Check if we also have the JSON version in the list
                      const hasJson = calcResults.plots.some(p => p.endsWith('mag_structure.json'));
                      if (hasJson) return null; // Skip rendering the PNG if JSON is present
                    }

                    return (
                      <div key={idx} className="card p-0 overflow-hidden shadow-lg">
                        <div className="p-sm glass border-b border-light flex-between">
                          <span className="font-bold text-sm uppercase tracking-wider opacity-70">
                            {(() => {
                              if (plotUrl.includes('disp')) return 'Spin Wave Dispersion';
                              if (plotUrl.includes('sqw')) return 'S(Q,ω) Intensity Map';
                              if (plotUrl.includes('powder')) return 'Powder Average';
                              if (plotUrl.includes('mag_structure')) return 'Magnetic Structure';
                              return 'Result Plot';
                            })()}
                          </span>
                          <a href={plotUrl} download className="icon-btn" title="Download Plot">
                            <Download size={14} />
                          </a>
                        </div>
                        <div className="plot-container bg-white">
                          {/* Add timestamp to bust cache */}
                          <img src={`${plotUrl}?t=${Date.now()}`} alt="Result Plot" className="w-full h-auto object-contain" />
                        </div>
                      </div>
                    )
                  })}
                  {calcResults.plots.length === 0 && (
                    <div className="card opacity-60 text-center py-xl">
                      <Info className="mx-auto mb-sm" />
                      <p>No plots generated. Enable plotting in "Tasks & Plotting" tab.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {!calcResults && !calcError && !calcLoading && (
              <div className="card glass opacity-40 text-center py-xxl border-dashed">
                <BarChart2 size={48} className="mx-auto mb-md opacity-30" />
                <p className="text-lg">Ready to Calculate</p>
                <p className="text-sm mt-sm max-w-md mx-auto">
                  Press the "Run Calculation" button to minimize energy (if enabled) and compute spin wave dispersion/intensity maps based on your configuration.
                </p>
              </div>
            )}

            {/* Log Console - Always visible in Run tab */}
            <LogConsole logs={logs} connected={wsConnected} />

          </div>
        )}

        {showVisualizer && (
          <>
            <div
              className="resizer"
              onMouseDown={startResizing('right')}
            ></div>
            <aside className="right-preview glass" style={{ width: visualizerWidth }}>
              <div className="preview-container relative">
                <Visualizer
                  atoms={previewAtoms}
                  bonds={bonds.filter(b => !hiddenBondLabels.has(getBondKey(b.rule_value || b.value)))}
                  lattice={config.lattice}
                  isDark={isDark}
                  onBondClick={setSelectedBond}
                  selectedBond={selectedBond}
                />

                {/* Visualizer Interaction Overlay Panel */}
                {/* Visualizer Interaction Overlay Panel */}
                {selectedBond && (
                  <div className="visualizer-overlay top-left animate-slide-in p-md card glass border-accent shadow-glow flex-col gap-sm" style={{ pointerEvents: 'auto', zIndex: 100, marginTop: '40px', maxWidth: '280px' }}>

                    {/* Header */}
                    <div className="flex-between align-center border-b border-color/20 pb-sm mb-xs">
                      <h4 className="font-bold text-accent m-0 text-sm">Selected Bond</h4>
                      <div
                        className="cursor-pointer text-muted hover:text-red-400 transition-colors opacity-50 hover:opacity-100"
                        onClick={() => setSelectedBond(null)}
                        title="Close"
                      >
                        <XCircle size={18} />
                      </div>
                    </div>

                    {/* Bond Info */}
                    <div className="bg-black/20 rounded p-xs flex-between align-center text-xs font-mono border border-color/10">
                      <div className="flex align-center gap-xs">
                        <span className="text-secondary font-bold">{previewAtoms[selectedBond.atom_i]?.label || selectedBond.atom_i}</span>
                        <ArrowDown size={10} className="rotate-[-90deg] opacity-50" />
                        <span className="text-secondary font-bold">{previewAtoms[selectedBond.atom_j]?.label || selectedBond.atom_j}</span>
                      </div>
                      <div className="opacity-60 text-xxs">
                        Offset: [{selectedBond.offset ? selectedBond.offset.join(',') : '0,0,0'}]
                      </div>
                    </div>

                    {/* Interactions List */}
                    <div className="flex-col gap-sm overflow-y-auto custom-scrollbar pr-xs" style={{ maxHeight: '250px' }}>
                      {(() => {
                        const matchingBonds = bonds.filter(b =>
                          b.atom_i === selectedBond.atom_i &&
                          b.atom_j === selectedBond.atom_j &&
                          (b.offset || []).join(',') === (selectedBond.offset || []).join(',')
                        );

                        if (matchingBonds.length === 0) {
                          return <div className="text-xs text-muted text-center italic py-2">No interactions</div>;
                        }

                        return matchingBonds.map((bond, idx) => {
                          const matrix = calculateExchangeMatrixSymbolic(bond, config.parameters);
                          return (
                            <div key={idx} className="p-xs bg-black/10 rounded border border-color/10">
                              <div className="flex align-center gap-xs mb-xs text-xs font-bold opacity-80">
                                {bond.type === 'heisenberg' && <Zap size={10} className="text-yellow-400" />}
                                {bond.type.includes('dm') && <Wind size={10} className="text-cyan-400" />}
                                {bond.type.includes('anisotropic') && <Crosshair size={10} className="text-purple-400" />}
                                <span className="capitalize">{bond.type.replace(/_/g, ' ')}</span>
                              </div>

                              {/* Matrix */}
                              {matrix ? (
                                <div className="exchange-matrix-grid" style={{ transform: 'scale(0.95)', transformOrigin: 'top left', width: '100%' }}>
                                  {matrix.flat().map((val, i) => (
                                    <div key={i} className={`exchange-matrix-cell ${val === 0 || val === '0' || val === '0.0' ? 'zero' : ''}`} style={{ fontSize: '9px', padding: '2px' }}>
                                      {typeof val === 'number' ? Number(val.toFixed(5)) : val}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-xs font-mono break-all opacity-70">{JSON.stringify(bond.value)}</div>
                              )}
                            </div>
                          );
                        });
                      })()}
                    </div>

                    {/* Footer Actions */}
                    <div className="relative pt-sm border-t border-color/20 mt-auto">
                      {interactionMenuOpen && (
                        <div className="absolute bottom-full left-0 right-0 mb-2 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl overflow-hidden flex flex-col p-1 z-50 animate-in fade-in zoom-in-95 duration-200">
                          <button
                            className="btn btn-ghost btn-xs justify-start gap-2 hover:bg-black/5 dark:hover:bg-white/10"
                            onClick={() => { addRuleFromVisualizer('heisenberg'); setInteractionMenuOpen(false); }}
                          >
                            <Zap size={14} className="text-amber-500" /> Heisenberg
                          </button>
                          <button
                            className="btn btn-ghost btn-xs justify-start gap-2 hover:bg-black/5 dark:hover:bg-white/10"
                            onClick={() => { addRuleFromVisualizer('dm'); setInteractionMenuOpen(false); }}
                          >
                            <Wind size={14} className="text-cyan-500" /> DM Interaction
                          </button>
                          <button
                            className="btn btn-ghost btn-xs justify-start gap-2 hover:bg-black/5 dark:hover:bg-white/10"
                            onClick={() => { addRuleFromVisualizer('anisotropic_exchange'); setInteractionMenuOpen(false); }}
                          >
                            <Crosshair size={14} className="text-purple-500" /> Anisotropic Exchange
                          </button>
                          <button
                            className="btn btn-ghost btn-xs justify-start gap-2 hover:bg-black/5 dark:hover:bg-white/10"
                            onClick={() => { addRuleFromVisualizer('interaction_matrix'); setInteractionMenuOpen(false); }}
                          >
                            <Box size={14} className="text-blue-500" /> Interaction Matrix
                          </button>
                          <button
                            className="btn btn-ghost btn-xs justify-start gap-2 hover:bg-black/5 dark:hover:bg-white/10"
                            onClick={() => { addRuleFromVisualizer('kitaev'); setInteractionMenuOpen(false); }}
                          >
                            <Box size={14} className="text-pink-500" /> Kitaev
                          </button>
                        </div>
                      )}
                      <button
                        className="btn btn-xs btn-primary w-full justify-between items-center group"
                        onClick={() => setInteractionMenuOpen(!interactionMenuOpen)}
                      >
                        <span className="flex items-center gap-2">
                          <Plus size={14} /> Add Interaction
                        </span>
                        <ChevronDown size={14} className={`transition-transform duration-200 ${interactionMenuOpen ? 'rotate-180' : ''}`} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </aside>
          </>
        )}
      </main>
      {
        notification && (
          <div className={`notification ${notification.type}`}>
            {notification.msg}
          </div>
        )
      }
    </div >
  )
}

export default App
