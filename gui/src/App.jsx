import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, Share2, Info, Magnet, Wind, Check, ChevronRight, Zap, Crosshair, FileText, BarChart2, Play, Image, ArrowDown, X, XCircle, Minus, ChevronDown } from 'lucide-react'
import yaml from 'js-yaml'
import Visualizer from './components/Visualizer'
import './App.css'

const LogConsole = ({ logs }) => {
  const endRef = React.useRef(null)
  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <div className="log-console mt-lg rounded-md border border-color bg-surface-hover p-md overflow-hidden flex flex-col" style={{ height: '300px' }}>
      <div className="flex-between mb-sm align-center border-b border-color pb-sm">
        <div className="flex align-center gap-xs text-xs font-mono opacity-70">
          <FileText size={14} />
          <span>Execution Logs</span>
        </div>
        <span className="text-xs opacity-50">{logs.length} lines</span>
      </div>
      <div className="flex-1 overflow-y-auto font-mono text-xs opacity-80 leading-relaxed custom-scrollbar">
        {logs.length === 0 ? (
          <div className="text-center opacity-30 mt-xl">Waiting for logs...</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="whitespace-pre-wrap break-all py-xxs">
              {log}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('structure')
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
  const [zFilter, setZFilter] = useState(false) // Filter for z=0 plane in 2D
  const [isAddingParam, setIsAddingParam] = useState(false)
  const [newParamName, setNewParamName] = useState('')
  const [calcLoading, setCalcLoading] = useState(false)
  const [calcResults, setCalcResults] = useState(null)

  const [calcError, setCalcError] = useState(null)

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

  // WebSocket Log Connection
  React.useEffect(() => {
    // Only connect if on run tab or globally? Let's do globally to catch background logs
    // But maybe retry if connection fails?
    let ws = null;
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      // Note: Vite proxy handles /ws -> localhost:8000
      const wsUrl = `${protocol}//${window.location.host}/ws/logs`;
      console.log("Connecting log WS:", wsUrl);

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("Log WebSocket Connected");
      };

      ws.onmessage = (event) => {
        setLogs(prev => {
          // Keep last 1000 lines
          const newLogs = [...prev, event.data];
          if (newLogs.length > 1000) return newLogs.slice(newLogs.length - 1000);
          return newLogs;
        });
      };

      ws.onclose = () => {
        console.log("Log WebSocket Closed");
      };

      ws.onerror = (err) => {
        console.error("Log WebSocket Error:", err);
      };

    } catch (e) {
      console.error("Failed to init WebSocket:", e);
    }

    return () => {
      if (ws) ws.close();
    }
  }, []);


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
    lattice: { a: 20.645, b: 8.383, c: 6.442, alpha: 90, beta: 90, gamma: 90, space_group: 43, dimensionality: '3D' },
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
    parameters: { S: 1.0, H_mag: 20.0, H_dir: [0, 0, 1], J1: 2.49, J2: 2.79, J3: 5.05, G1: 0.28, Dx: 2.67, D: 0.0 },
    tasks: {
      run_minimization: true,
      run_dispersion: true,
      calculate_dispersion_new: true,
      plot_dispersion: true,
      run_sqw_map: true,
      calculate_sqw_map_new: true,
      plot_sqw_map: true,
      export_csv: false
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
      num_starts: 1000,
      n_workers: 8,
      early_stopping: 10,
      method: "L-BFGS-B"
    },
    calculation: {
      cache_mode: 'none'
    }
  }

  const [config, setConfig] = useState(() => {
    try {
      const saved = localStorage.getItem('magcalc_config');
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.error("Failed to load config from localStorage", e);
    }
    return DEMO_CONFIG;
  })

  // Persistence Effect
  React.useEffect(() => {
    localStorage.setItem('magcalc_config', JSON.stringify(config));
  }, [config]);

  // Symmetry Expansion Effect for Visualizer
  React.useEffect(() => {
    const updatePreview = async () => {
      try {
        const payload = {
          data: {
            crystal_structure: {
              lattice_parameters: config.lattice,
              wyckoff_atoms: config.wyckoff_atoms,
              atom_mode: atomMode,
              dimensionality: config.lattice.dimensionality
            },
            interactions: {
              symmetry_rules: config.symmetry_interactions,
              list: interactionMode === 'explicit' ? config.explicit_interactions : undefined,
              single_ion_anisotropy: config.single_ion_anisotropy || []
            },
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
  }, [config.lattice, config.wyckoff_atoms, atomMode, config.lattice.dimensionality, config.symmetry_interactions, config.explicit_interactions, config.parameters, interactionMode])

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

  const DEFAULT_CONFIG = {
    lattice: { a: 5.0, b: 5.0, c: 5.0, alpha: 90, beta: 90, gamma: 90, space_group: 1, dimensionality: '3D' },
    wyckoff_atoms: [],
    magnetic_elements: ["Cu"],
    symmetry_interactions: [],
    explicit_interactions: [],
    parameters: { S: 1.0, H_mag: 0.0, H_dir: [0, 0, 1] },
    tasks: {
      minimization: true,
      dispersion: true,
      sqw_map: true,
      export_csv: false
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
      num_starts: 1000,
      n_workers: 8,
      early_stopping: 10,
      method: "L-BFGS-B"
    },
    calculation: {
      cache_mode: 'none'
    }
  }

  const resetToDefaults = () => {
    if (window.confirm("Are you sure you want to load the default example (aCVO)?\nCurrent changes will be lost.")) {
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
            newConfig.lattice.lattice_vectors = doc.crystal_structure.lattice_vectors
          }
          if (doc.crystal_structure.dimensionality) {
            newConfig.lattice.dimensionality = doc.crystal_structure.dimensionality
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
        if (doc.tasks) newConfig.tasks = { ...newConfig.tasks, ...doc.tasks }
        if (doc.plotting) newConfig.plotting = { ...newConfig.plotting, ...doc.plotting }
        if (doc.minimization) newConfig.minimization = { ...newConfig.minimization, ...doc.minimization }
        if (doc.calculation) newConfig.calculation = { ...newConfig.calculation, ...doc.calculation }
        if (doc.magnetic_structure) newConfig.magnetic_structure = { ...newConfig.magnetic_structure, ...doc.magnetic_structure }
        if (doc.q_path) {
          const { path, points_per_segment, ...points } = doc.q_path
          newConfig.q_path = {
            points: points || {},
            path: path || [],
            points_per_segment: points_per_segment || 100
          }
        }
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
    // Structure the input for the Expansion Backend
    const input = {
      crystal_structure: {
        lattice_parameters: config.lattice,
        wyckoff_atoms: config.wyckoff_atoms,
        atom_mode: atomMode,
        magnetic_elements: config.magnetic_elements || ["Cu"],
        dimensionality: config.lattice.dimensionality === '2D' ? 2 : (config.lattice.dimensionality === '3D' ? 3 : config.lattice.dimensionality)
      },
      interactions: interactionMode === 'explicit' ? { list: config.explicit_interactions || [] } : {
        symmetry_rules: config.symmetry_interactions
      },
      magnetic_structure: config.magnetic_structure,
      parameters: config.parameters,
      tasks: {
        ...config.tasks,
        calculate_dispersion_new: config.tasks.run_dispersion,
        calculate_sqw_map_new: config.tasks.run_sqw_map
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
        enabled: config.tasks.run_minimization,
        ...config.minimization
      },
      output: config.output
    }

    try {
      // Use the design config directly for export (cleaner YAML)
      // instead of the expanded calculation-ready config.
      let expanded = input;

      // try {
      //   console.log('Fetching expanded config for export...')
      //   const response = await fetch('/api/expand-config', { ... })
      //   if (!response.ok) throw new Error(...)
      //   expanded = await response.json()
      // } catch (err) { ... } -> We skip this now.

      // We can just proceed with 'input' as 'expanded'
      console.log('Generating design YAML file...')

      // Conditionally omit magnetic structure
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
              dimensionality: config.lattice.dimensionality,
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
      distance: orbit.distance,
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

  const updateField = (section, field, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: { ...prev[section], [field]: value }
    }))
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

    const newRule = {
      type: type,
      ref_pair: [previewAtoms[selectedBond.atom_i]?.label || "?", previewAtoms[selectedBond.atom_j]?.label || "?"],
      offset: selectedBond.offset || [0, 0, 0],
      distance: selectedBond.distance || 0.0,
      value: type === 'heisenberg' ? 'J0' : (type === 'dm' ? ['D1', 'D2', 'D3'] : ['G1', 'G2', 'G3'])
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
          value: type === 'heisenberg' ? 'J0' : (type === 'dm' ? ['D1', 'D2', 'D3'] : ['G1', 'G2', 'G3'])
        }]
      }));
    }
    showNotify(`Added ${type} interaction`, 'success');
  }

  const runCalculation = async () => {
    setLogs([])
    setCalcLoading(true)
    setCalcResults(null)
    setCalcError(null)

    // Construct payload as expected by expand-config logic backend
    const input = {
      crystal_structure: {
        lattice_parameters: config.lattice,
        wyckoff_atoms: config.wyckoff_atoms,
        atom_mode: atomMode,
        dimensionality: [2, '2D', '2'].includes(config.lattice.dimensionality) ? 2 : 3,
        magnetic_elements: config.magnetic_elements || ["Cu"]
      },
      interactions: interactionMode === 'explicit' ? { list: config.explicit_interactions || [] } : {
        symmetry_rules: config.symmetry_interactions
      },
      magnetic_structure: config.magnetic_structure,
      parameters: config.parameters,
      tasks: {
        ...config.tasks,
        calculate_dispersion_new: config.tasks.run_dispersion,
        calculate_sqw_map_new: config.tasks.run_sqw_map
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
        enabled: config.tasks.run_minimization,
        ...config.minimization
      }
    }

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
      setCalcError(err.message)
      showNotify("Calculation failed", "error")
    } finally {
      setCalcLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="background-glow"></div>

      <header className="glass">
        <div className="logo animate-fade-in">
          <div className="icon-wrapper gradient-bg">
            <img src="/spin_vector_icon.png" alt="Spin Vector Icon" className="w-full h-full object-cover" />
          </div>
          <div>
            <h1 className="header-title">pyMagCalc Studio</h1>
            <div className="flex-gap-xs align-center">
              <span className="subtitle">Configure Models & Calculate Spin-Waves</span>
            </div>
          </div>
        </div>
        <div className="header-actions">
          <label className="btn btn-secondary glass cursor-pointer">
            <Share2 size={16} /> Load CIF
            <input type="file" accept=".cif" hidden onChange={handleCifUpload} />
          </label>
          <label className="btn btn-secondary glass cursor-pointer">
            <Code size={16} /> Load YAML
            <input type="file" accept=".yaml,.yml" hidden onChange={handleImport} />
          </label>
          <button className="btn btn-secondary glass cursor-pointer" onClick={resetToDefaults}>
            <Trash2 size={16} /> Load Defaults
          </button>
          <button className="btn btn-primary shadow-glow" onClick={handleExportYaml}>
            <Download size={16} /> Export YAML
          </button>
        </div>
      </header>

      <main>
        <aside className="sidebar glass" style={{ width: sidebarWidth }}>
          <nav>
            <button className={`nav-item ${activeTab === 'structure' ? 'active' : ''}`} onClick={() => setActiveTab('structure')}>
              <Box size={20} /> Structure
            </button>
            <button className={`nav-item ${activeTab === 'interactions' ? 'active' : ''}`} onClick={() => setActiveTab('interactions')}>
              <Magnet size={20} /> Interactions
            </button>
            <button className={`nav-item ${activeTab === 'params' ? 'active' : ''}`} onClick={() => setActiveTab('params')}>
              <Settings size={20} /> Environment
            </button>
            <button className={`nav-item ${activeTab === 'tasks' ? 'active' : ''}`} onClick={() => setActiveTab('tasks')}>
              <Activity size={20} /> Tasks & Plotting
            </button>
            <button className={`nav-item ${activeTab === 'magstruct' ? 'active' : ''}`} onClick={() => setActiveTab('magstruct')}>
              <Wind size={20} /> Mag. Structure
            </button>
            <div className="nav-divider"></div>
            <button className={`nav-item ${activeTab === 'run' ? 'active' : ''}`} onClick={() => setActiveTab('run')}>
              <BarChart2 size={20} /> Run & Analyze
            </button>
          </nav>


        </aside>

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
                      <label>Space Group (#)</label>
                      <input type="number" value={config.lattice.space_group} className="minimal-input"
                        onChange={(e) => updateField('lattice', 'space_group', parseInt(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Dimensionality</label>
                      <select
                        className="minimal-select"
                        value={config.lattice.dimensionality || '3D'}
                        onChange={(e) => {
                          const val = e.target.value;
                          updateField('lattice', 'dimensionality', val);
                          if (val === '2D') {
                            updateField('lattice', 'alpha', 90);
                            updateField('lattice', 'beta', 90);
                          }
                        }}
                      >
                        <option value="3D">3D (Bulk)</option>
                        <option value="2D">2D (Monolayer/Layered)</option>
                      </select>
                    </div>
                  </div>

                  {config.lattice.dimensionality === '2D' && (
                    <div className="mt-md p-md glass shadow-sm rounded-xl border border-blue-500/20 animate-fade-in">
                      <div className="flex align-center gap-xs text-xs font-bold text-blue-400 mb-xs">
                        <Info size={14} />
                        <span>Note on 2D Symmetry</span>
                      </div>
                      <p className="text-xxs opacity-70 leading-relaxed">
                        Symmetry operations (like the glide in SG 163) may generate multiple planes in a single unit cell.
                        If processing a monolayer, consider using a non-glide space group or <strong>Explicit Unit Cell</strong> mode.
                      </p>
                    </div>
                  )}
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
                            <button onClick={() => {
                              const next = config.symmetry_interactions.filter((_, i) => i !== idx); setConfig({ ...config, symmetry_interactions: next })
                            }} className="icon-btn text-error"><Trash2 size={14} /></button>
                          </div>

                          <div className="interaction-params">
                            <div className="input-group">
                              <label>Distance (Å)</label>
                              <input type="number" step="0.01" className="minimal-input" value={inter.distance} onChange={(e) => {
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
                                <div className="grid grid-cols-3 gap-1 bg-black/20 p-xs rounded border border-color/30">
                                  {inter.value.map((row, r) => row.map((cell, c) => (
                                    <input
                                      key={`${r}-${c}`}
                                      type="text"
                                      className={`text-center text-xs p-1 bg-transparent border-none outline-none w-full ${cell === '0' || cell === '0.0' ? 'opacity-30' : 'text-accent font-bold'}`}
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
                                  setConfig({ ...config, symmetry_interactions: next })
                                }}
                              >
                                <option value="heisenberg">Heisenberg</option>
                                <option value="dm">DM Interaction</option>
                                <option value="anisotropic_exchange">Anisotropic</option>
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
                                    <div key={i} className={`exchange-matrix-cell ${val === 0 ? 'zero' : ''}`}>
                                      {typeof val === 'number' ? val.toFixed(3) : val}
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
                                    distance: n.distance,
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
                            <button onClick={() => {
                              const next = config.explicit_interactions.filter((_, i) => i !== idx);
                              setConfig({ ...config, explicit_interactions: next });
                            }} className="icon-btn text-error"><Trash2 size={14} /></button>
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
                              <input type="number" step="0.01" className="minimal-input" value={inter.distance} onChange={(e) => {
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
                            <Eye size={18} />
                          </div>
                          <div className="task-info">
                            <span className="task-name">S(Q,w) Map</span>
                            <span className="task-desc">Calculate & Plot</span>
                          </div>
                          <div className="task-check">
                            <Check size={12} strokeWidth={4} />
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="card shadow-glow">
                      <h3>Display Parameters</h3>
                      <div className="grid-form mt-md">
                        <div className="input-group">
                          <label>Energy Min (meV)</label>
                          <input type="number" step="0.1" value={config.plotting.energy_min} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'energy_min', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Energy Max (meV)</label>
                          <input type="number" step="0.1" value={config.plotting.energy_max} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'energy_max', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Broadening (meV)</label>
                          <input type="number" step="0.01" value={config.plotting.broadening} className="minimal-input"
                            onChange={(e) => updateField('plotting', 'broadening', parseFloat(e.target.value))} />
                        </div>
                        <div className="input-group">
                          <label>Energy Res. (meV)</label>
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

                  <div className="card shadow-glow">
                    <h3>Minimization Parameters</h3>
                    <div className="grid-form mt-md">
                      <div className="input-group">
                        <label>Num Starts</label>
                        <input type="number" value={config.minimization.num_starts} className="minimal-input"
                          onChange={(e) => updateField('minimization', 'num_starts', parseInt(e.target.value))} />
                      </div>
                      <div className="input-group">
                        <label>N Workers</label>
                        <input type="number" value={config.minimization.n_workers} className="minimal-input"
                          onChange={(e) => updateField('minimization', 'n_workers', parseInt(e.target.value))} />
                      </div>
                      <div className="input-group">
                        <label>Early Stopping</label>
                        <input type="number" value={config.minimization.early_stopping} className="minimal-input"
                          onChange={(e) => updateField('minimization', 'early_stopping', parseInt(e.target.value))} />
                      </div>
                      <div className="input-group">
                        <label>Method</label>
                        <select
                          className="minimal-select"
                          value={config.minimization.method}
                          onChange={(e) => updateField('minimization', 'method', e.target.value)}
                        >
                          <option value="L-BFGS-B">L-BFGS-B</option>
                          <option value="TNC">TNC</option>
                          <option value="SLSQP">SLSQP</option>
                        </select>
                      </div>
                    </div>

                    <h3 className="mt-lg">Calculation Settings</h3>
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
          </section>
        )}

        {activeTab === 'run' && (
          <div className="content-area animate-fade-in full-width-tab">
            <div className="flex-between align-center mb-xl">
              <div>
                <h2 className="section-title mb-xs">Run Calculation & Analysis</h2>
                <p className="text-sm opacity-60">Execute the simulation using current settings and visualize results.</p>
              </div>
              <button
                className={`btn btn-primary btn-lg shadow-glow ${calcLoading ? 'opacity-50 pointer-events-none' : ''}`}
                onClick={runCalculation}
              >
                {calcLoading ? (
                  <>
                    <Activity className="animate-spin mr-sm" /> Calculating...
                  </>
                ) : (
                  <>
                    <Play size={18} className="mr-sm" /> Run Calculation
                  </>
                )}
              </button>
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
                  {calcResults.plots.map((plotUrl, idx) => (
                    <div key={idx} className="card p-0 overflow-hidden shadow-lg">
                      <div className="p-sm glass border-b border-light flex-between">
                        <span className="font-bold text-sm uppercase tracking-wider opacity-70">
                          {plotUrl.includes('disp') ? 'Spin Wave Dispersion' : 'S(Q,ω) Intensity Map'}
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
                  ))}
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
            <LogConsole logs={logs} />

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
                  lattice={config.lattice}
                  isDark={isDark}
                  dimensionality={config.lattice.dimensionality}
                  zFilter={zFilter}
                  bonds={bonds}
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
                                      {val}
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
                {config.lattice.dimensionality === '2D' && (
                  <div className="visualizer-overlay bottom-right">
                    <button
                      className={`btn btn-xs ${zFilter ? 'btn-primary' : 'btn-secondary glass'}`}
                      onClick={() => setZFilter(!zFilter)}
                      title="Filter to show only the Z=0 atomic plane"
                    >
                      <Eye size={12} className="mr-xs" />
                      {zFilter ? "Show All Planes" : "Show Only Z=0"}
                    </button>
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

const calculateExchangeMatrix = (inter, numericParams = {}) => {
  const type = inter.type;
  const value = inter.value;

  // Helper to safely evaluate parameter string
  const evalParam = (val) => {
    if (typeof val === 'number') return val;
    if (!val) return 0.0;
    if (numericParams[val] !== undefined) return numericParams[val];
    // Try simple eval if string is number
    const f = parseFloat(val);
    if (!isNaN(f)) return f;
    return 0.0;
  };

  try {
    if (type === 'heisenberg') {
      const j = evalParam(value);
      return [
        [j, 0, 0],
        [0, j, 0],
        [0, 0, j]
      ];
    } else if (type === 'dm' || type === 'dm_interaction') {
      // DM value should be array [Dx, Dy, Dz] or string "Dx,Dy,Dz"
      let vec = [0, 0, 0];
      if (Array.isArray(value)) {
        vec = value.map(evalParam);
      } else if (typeof value === 'string') {
        vec = value.split(',').map(s => evalParam(s.trim()));
      }

      const [dx, dy, dz] = vec;
      // Skew symmetric
      return [
        [0, dz, -dy],
        [-dz, 0, dx],
        [dy, -dx, 0]
      ];
    } else if (type === 'anisotropic' || type === 'anisotropic_exchange') {
      // Anisotropic value should be array [Jx, Jy, Jz] or string
      let vec = [0, 0, 0];
      if (Array.isArray(value)) {
        vec = value.map(evalParam);
      } else if (typeof value === 'string') {
        vec = value.split(',').map(s => evalParam(s.trim()));
      }
      const [jx, jy, jz] = vec;
      return [
        [jx, 0, 0],
        [0, jy, 0],
        [0, 0, jz]
      ];
    }
  } catch (e) {
    console.error("Matrix Calc Error", e);
    return null;
  }
  return null;
};

const calculateExchangeMatrixSymbolic = (inter, numericParams = {}) => {
  const type = inter.type;
  const value = inter.value;

  // Helper to get symbol or value
  const getSymbol = (val) => {
    if (val === undefined || val === null) return 0;
    if (typeof val === 'number') return val;
    let s = String(val).trim();

    // Clean up symbolic math artifacts from backend
    // Remove leading "1.0*" or "1*"
    if (s.startsWith('1.0*')) s = s.substring(4);
    else if (s.startsWith('1*')) s = s.substring(2);

    // Replace leading "-1.0*" or "-1*" with "-"
    if (s.startsWith('-1.0*')) s = '-' + s.substring(5);
    else if (s.startsWith('-1*')) s = '-' + s.substring(3);

    if (s === '0' || s === '0.0') return 0;

    // If it looks like a pure number, parse it but keep 0.0 as 0
    // But be careful not to parse "J1" as NaN -> string
    if (!isNaN(parseFloat(s)) && isFinite(s)) {
      if (parseFloat(s) === 0) return 0;
      // If it became a simple number like "2.0", maybe return number?
      // But user wants symbols. If it is "1.5", return "1.5".
      return s;
    }
    return s;
  };

  try {
    if (type === 'heisenberg') {
      const j = getSymbol(value);
      return [
        [j, 0, 0],
        [0, j, 0],
        [0, 0, j]
      ];
    } else if (type === 'dm' || type === 'dm_interaction') {
      // DM value should be array or string
      let vec = [0, 0, 0];
      if (Array.isArray(value)) {
        vec = value.map(getSymbol);
      } else if (typeof value === 'string') {
        vec = value.split(',').map(getSymbol);
      }

      const [dx, dy, dz] = vec;

      // Handle negation for symbols
      const neg = (v) => {
        if (v === 0) return 0;
        if (typeof v === 'string') {
          // Handle double negatives
          if (v.startsWith('-')) return v.substring(1);
          return `-${v}`;
        }
        return -v;
      }

      return [
        [0, dz, neg(dy)],
        [neg(dz), 0, dx],
        [dy, neg(dx), 0]
      ];
    } else if (type === 'anisotropic' || type === 'anisotropic_exchange') {
      let vec = [0, 0, 0];
      if (Array.isArray(value)) {
        vec = value.map(getSymbol);
      } else if (typeof value === 'string') {
        vec = value.split(',').map(getSymbol);
      }
      const [jx, jy, jz] = vec;
      return [
        [jx, 0, 0],
        [0, jy, 0],
        [0, 0, jz]
      ];
    }
  } catch (e) {
    console.error("Matrix Calc Error", e);
    return null;
  }
  return null;
};

export default App
