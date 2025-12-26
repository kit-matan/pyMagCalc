import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, Share2, Info, Magnet, Wind, Check, ChevronRight, Zap, Crosshair, FileText, BarChart2, Play } from 'lucide-react'
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
  const [interactionMode, setInteractionMode] = useState('explicit') // 'symmetry' or 'explicit'
  const [atomMode, setAtomMode] = useState('explicit') // 'symmetry' or 'explicit'
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

  const [config, setConfig] = useState({
    lattice: { a: 7.33, b: 7.33, c: 17.1374, alpha: 90, beta: 90, gamma: 120, space_group: 163, dimensionality: '2D' },
    wyckoff_atoms: [
      { label: 'Fe0', pos: [0.0, 0.0, 0.0], spin_S: 2.5 },
      { label: 'Fe1', pos: [0.5, 0.0, 0.0], spin_S: 2.5 },
      { label: 'Fe2', pos: [0.0, 0.5, 0.0], spin_S: 2.5 }
    ],
    symmetry_interactions: [
      { type: 'heisenberg', ref_pair: ['Fe0', 'Fe1'], distance: 3.665, value: 'J1' },
      { type: 'heisenberg', ref_pair: ['Fe0', 'Fe1'], distance: 6.348, value: 'J2' },
      { type: 'dm', ref_pair: ['Fe1', 'Fe2'], distance: 3.665, value: ['0', '-Dy', '-Dz'] }
    ],
    explicit_interactions: [
      { type: 'heisenberg', distance: 3.665, value: "J1" },
      { type: 'heisenberg', distance: 6.348, value: "J2" },
      { type: 'dm_manual', atom_i: 0, atom_j: 1, offset_j: [0, 0, 0], value: ["0", "-Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 0, atom_j: 2, offset_j: [0, 0, 0], value: ["-0.86602540378*Dy", "-0.5*Dy", "Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 0, atom_j: 1, offset_j: [-1, 0, 0], value: ["0", "-Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 0, atom_j: 2, offset_j: [0, -1, 0], value: ["-0.86602540378*Dy", "-0.5*Dy", "Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 1, atom_j: 0, offset_j: [0, 0, 0], value: ["0", "Dy", "Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 1, atom_j: 2, offset_j: [0, -1, 0], value: ["-0.86602540378*Dy", "0.5*Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 1, atom_j: 0, offset_j: [1, 0, 0], value: ["0", "Dy", "Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 1, atom_j: 2, offset_j: [1, 0, 0], value: ["-0.86602540378*Dy", "0.5*Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 2, atom_j: 0, offset_j: [0, 0, 0], value: ["0.86602540378*Dy", "0.5*Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 2, atom_j: 1, offset_j: [-1, 0, 0], value: ["0.86602540378*Dy", "-0.5*Dy", "Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 2, atom_j: 0, offset_j: [0, 1, 0], value: ["0.86602540378*Dy", "0.5*Dy", "-Dz"], distance: 3.665 },
      { type: 'dm_manual', atom_i: 2, atom_j: 1, offset_j: [0, 1, 0], value: ["0.86602540378*Dy", "-0.5*Dy", "Dz"], distance: 3.665 }
    ],
    parameters: { S: 2.5, H_mag: 0.0, H_dir: [0, 0, 1], J1: 3.23, J2: 0.11, Dy: 0.218, Dz: -0.195 },
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
      points: { Gamma: [0, 0, 0], M: [0.5, 0, 0], K: [0.333, 0.333, 0] },
      path: ['Gamma', 'M', 'K', 'Gamma'],
      points_per_segment: 100
    },
    plotting: {
      energy_min: 0,
      energy_max: 20,
      broadening: 0.2,
      energy_resolution: 0.05,
      momentum_max: 4.0,
      save_plot: true
    },
    output: {
      disp_csv_filename: 'disp_data.csv',
      sqw_csv_filename: 'sqw_data.csv'
    },
    magnetic_structure: {
      enabled: false,
      type: 'pattern',
      pattern_type: 'antiferromagnetic',
      directions: [
        [-1, 0, 0],
        [0.5, -0.86602540378, 0],
        [0.5, 0.86602540378, 0]
      ]
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
  })

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
              list: interactionMode === 'explicit' ? config.explicit_interactions : undefined
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

  const handleImport = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const doc = yaml.load(event.target.result)
        const newConfig = { ...config }

        // Pure Model check (symmetry_rules)
        if (doc.interactions && doc.interactions.symmetry_rules) {
          newConfig.symmetry_interactions = doc.interactions.symmetry_rules
        }

        if (doc.crystal_structure) {
          if (doc.crystal_structure.lattice_parameters) {
            newConfig.lattice = { ...newConfig.lattice, ...doc.crystal_structure.lattice_parameters }
          }
          if (doc.crystal_structure.wyckoff_atoms) {
            newConfig.wyckoff_atoms = doc.crystal_structure.wyckoff_atoms.map(a => ({
              label: a.label || 'Atom',
              pos: a.pos || [0, 0, 0],
              spin_S: a.spin_S || 0.5
            }))
          }
        }
        if (doc.parameters) newConfig.parameters = { ...newConfig.parameters, ...doc.parameters }
        if (doc.tasks) newConfig.tasks = { ...newConfig.tasks, ...doc.tasks }
        if (doc.plotting) newConfig.plotting = { ...newConfig.plotting, ...doc.plotting }
        if (doc.q_path) {
          const { path, points_per_segment, ...points } = doc.q_path
          newConfig.q_path = {
            points: points || {},
            path: path || [],
            points_per_segment: points_per_segment || 100
          }
        }
        setConfig(newConfig)
        alert('Configuration imported successfully! Note: Symmetry rules may need manual adjustment.')
      } catch (err) {
        alert('Error parsing YAML: ' + err.message)
      }
    }
    reader.readAsText(file)
  }

  const handleExportYaml = async () => {
    // Structure the input for the Expansion Backend
    const input = {
      crystal_structure: {
        lattice_parameters: config.lattice,
        wyckoff_atoms: config.wyckoff_atoms,
        atom_mode: atomMode,
        dimensionality: config.lattice.dimensionality === '2D' ? 2 : (config.lattice.dimensionality === '3D' ? 3 : config.lattice.dimensionality)
      },
      interactions: interactionMode === 'explicit' ? { list: config.explicit_interactions || [] } : {
        symmetry_rules: config.symmetry_interactions
      },
      magnetic_structure: config.magnetic_structure,
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
        enabled: config.tasks.run_minimization,
        ...config.minimization
      }
    }

    try {
      console.log('Fetching expanded config for export...')
      const response = await fetch('/api/expand-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: input }),
      })

      if (!response.ok) throw new Error(`Server returned ${response.status}`)
      const expanded = await response.json()

      // Conditionally omit magnetic structure
      if (!config.magnetic_structure.enabled) {
        delete expanded.magnetic_structure;
      }

      console.log('Expansion successful, generating file...')
      const data = yaml.dump(expanded)

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
        dimensionality: config.lattice.dimensionality
      },
      interactions: interactionMode === 'explicit' ? { list: config.explicit_interactions || [] } : {
        symmetry_rules: config.symmetry_interactions
      },
      magnetic_structure: config.magnetic_structure,
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
                    <h2 className="section-title compact">Bonding Rules</h2>

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
                                <span className="interaction-type">{inter.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
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
                              <label>Value (Symbol)</label>
                              <input type="text" className="minimal-input accent-text" value={inter.value} onChange={(e) => {
                                const next = [...config.symmetry_interactions]; next[idx].value = e.target.value; setConfig({ ...config, symmetry_interactions: next })
                              }} />
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
                              </select>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="mt-md">
                      <h3 className="section-title text-sm mb-sm">Neighbor Shell Suggestions</h3>
                      <div className="flex gap-2 mb-sm items-center">
                        <button className="btn btn-secondary btn-sm" onClick={fetchNeighbors}>
                          <Activity size={14} className="mr-xs" />
                          Re-calculate Neighbors
                        </button>
                      </div>
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
                                {inter.type === 'heisenberg' ? <Zap size={16} /> : <Wind size={16} />}
                              </div>
                              <div>
                                <span className="interaction-type">{inter.type === 'heisenberg' ? 'Heisenberg' : 'DM Manual'}</span>
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
                                if (e.target.value.startsWith('dm')) {
                                  next[idx].value = ["0", "0", "0"];
                                } else {
                                  next[idx].value = "J1";
                                }
                                setConfig({ ...config, explicit_interactions: next });
                              }}>
                                <option value="heisenberg">Heisenberg</option>
                                <option value="dm_manual">DM Manual</option>
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
                                  {['Dx', 'Dy', 'Dz'].map((label, k) => (
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
                    <div className="input-group">
                      <label>Default Spin (S)</label>
                      <input type="number" step="0.5" value={config.parameters.S} className="minimal-input" onChange={(e) => updateField('parameters', 'S', parseFloat(e.target.value))} />
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
                <div className="grid-2 mt-md">
                  <div className="card shadow-glow">
                    <h3>Calculation Tasks</h3>
                    <div className="task-cards-grid">
                      {Object.keys(config.tasks).filter(k => k !== 'export_csv').map(taskKey => {
                        const Icon = taskKey.includes('plot') ? Eye : (taskKey.includes('minimization') ? Magnet : Activity);
                        const label = taskKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        const desc = taskKey.includes('run') ? 'Calculate results' : 'Generate visualization';

                        return (
                          <div
                            key={taskKey}
                            className={`task-card ${config.tasks[taskKey] ? 'active' : ''}`}
                            onClick={() => updateField('tasks', taskKey, !config.tasks[taskKey])}
                          >
                            <div className="task-icon-box">
                              <Icon size={18} />
                            </div>
                            <div className="task-info">
                              <span className="task-name">{label}</span>
                              <span className="task-desc">{desc}</span>
                            </div>
                            <div className="task-check">
                              <Check size={12} strokeWidth={4} />
                            </div>
                          </div>
                        );
                      })}
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

                    <h3 className="mt-xl">Calculation Settings</h3>
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
                          'None' is recommended for small systems or when debugging symmetry.
                        </p>
                      </div>
                    </div>

                    <h3 className="mt-xl">Data Export</h3>
                    <div className="mt-md">
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

                <div className="grid-2 mt-md">
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
                      <div className="input-group">
                        <label>Momentum Max (Å⁻¹)</label>
                        <input type="number" step="0.1" value={config.plotting.momentum_max} className="minimal-input"
                          onChange={(e) => updateField('plotting', 'momentum_max', parseFloat(e.target.value))} />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card mt-xl">
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
                />
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
    </div>
  )
}

export default App
