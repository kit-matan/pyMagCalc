import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, Share2, Info, Magnet, Wind } from 'lucide-react'
import yaml from 'js-yaml'
import Visualizer from './components/Visualizer'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('structure')
  const [showVisualizer, setShowVisualizer] = useState(true)
  const [notification, setNotification] = useState(null)
  const [neighborDistances, setNeighborDistances] = useState([])
  const [interactionMode, setInteractionMode] = useState('explicit') // 'symmetry' or 'explicit'

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
      run_minimization: false,
      run_dispersion: true,
      plot_dispersion: true,
      run_sqw_map: true,
      plot_sqw_map: true
    },
    q_path: {
      points: { Gamma: [0, 0, 0], M: [0.5, 0, 0], K: [0.333, 0.333, 0] },
      path: ['Gamma', 'M', 'K', 'Gamma'],
      points_per_segment: 100
    },
    plotting: {
      energy_min: 0,
      energy_max: 10,
      broadening: 0.2,
      energy_resolution: 0.05,
      momentum_max: 4.0,
      save_plot: true
    },
    magnetic_structure: {
      enabled: true,
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
    }
  })

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
              dimensionality: config.lattice.dimensionality
            }
          }
        }),
      })
      if (!response.ok) throw new Error('Failed to fetch neighbors')
      const data = await response.json()
      setNeighborDistances(data)
    } catch (err) {
      console.error('Error fetching neighbors:', err)
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

  return (
    <div className="app-container">
      <div className="background-glow"></div>

      <header className="glass">
        <div className="logo animate-fade-in">
          <div className="icon-wrapper gradient-bg">
            <Beaker className="accent-icon" />
          </div>
          <div>
            <h1 className="vibrant-text">MagCalc Pure Designer</h1>
            <div className="flex-gap-xs align-center">
              <span className="version-tag">STABLE v2.0</span>
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
        <aside className="left-sidebar glass">
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
          </nav>

          <div className="mt-xl card glass text-xs opacity-60">
            <div className="flex-gap-sm mb-sm"><Info size={12} /> <b>Pure Design Mode</b></div>
            <p>Define the minimal set of basis atoms and interaction rules. Symmetry does the rest.</p>
          </div>
        </aside>

        <section className="content-area animate-fade-in">
          {activeTab === 'structure' && (
            <div className="form-section">
              <h2 className="section-title">Crystal Architecture</h2>
              <div className="card shadow-glow">
                <h3>Lattice Parameters</h3>
                <div className="grid-form mt-md">
                  {['a', 'b', 'c', 'alpha', 'beta', 'gamma'].map(k => (
                    <div key={k} className="input-group">
                      <label>{k}</label>
                      <input type="number" step="0.001" value={config.lattice[k]}
                        onChange={(e) => updateField('lattice', k, parseFloat(e.target.value))} />
                    </div>
                  ))}
                  <div className="input-group">
                    <label>Space Group (#)</label>
                    <input type="number" value={config.lattice.space_group}
                      onChange={(e) => updateField('lattice', 'space_group', parseInt(e.target.value))} />
                  </div>
                  <div className="input-group">
                    <label>Dimensionality</label>
                    <select
                      className="minimal-select"
                      value={config.lattice.dimensionality || '3D'}
                      onChange={(e) => updateField('lattice', 'dimensionality', e.target.value)}
                    >
                      <option value="3D">3D (Bulk)</option>
                      <option value="2D">2D (Monolayer/Layered)</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="card mt-xl">
                <div className="flex-between mb-md">
                  <h3>Basis Atoms (Wyckoff)</h3>
                  <button className="btn btn-secondary btn-sm" onClick={() => {
                    const next = [...config.wyckoff_atoms];
                    next.push({ label: 'Cu', pos: [0, 0, 0], spin_S: 0.5 });
                    setConfig({ ...config, wyckoff_atoms: next });
                  }}><Plus size={14} /> Add Basis</button>
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
                        <td><input type="text" className="mono" value={atom.label} onChange={(e) => {
                          const next = [...config.wyckoff_atoms]; next[idx].label = e.target.value; setConfig({ ...config, wyckoff_atoms: next })
                        }} /></td>
                        <td>
                          <div className="flex-gap-xs">
                            {[0, 1, 2].map(i => (
                              <input key={i} type="number" step="0.01" value={atom.pos[i]} onChange={(e) => {
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
              <div className="flex-center mb-lg">
                <div className="toggle-group glass p-xs rounded-full flex gap-xs">
                  <button
                    className={`btn px-md py-xs rounded-full transition-all ${interactionMode === 'symmetry' ? 'bg-accent text-white shadow-glow' : 'opacity-60 hover:opacity-100'}`}
                    onClick={() => setInteractionMode('symmetry')}
                  >
                    Symmetry Rules
                  </button>
                  <button
                    className={`btn px-md py-xs rounded-full transition-all ${interactionMode === 'explicit' ? 'bg-accent text-white shadow-glow' : 'opacity-60 hover:opacity-100'}`}
                    onClick={() => setInteractionMode('explicit')}
                  >
                    Explicit Interactions
                  </button>
                </div>
              </div>

              {interactionMode === 'symmetry' ? (
                <>
                  <div className="flex-between mb-md">
                    <h2 className="section-title">Bonding Rules</h2>
                    <button className="btn btn-primary btn-sm" onClick={() => {
                      const nextRules = [...config.symmetry_interactions, { type: 'heisenberg', distance: 3.0, value: 'J1' }];
                      const nextParams = { ...config.parameters };
                      if (!nextParams.J1) nextParams.J1 = 0.0;
                      setConfig({ ...config, symmetry_interactions: nextRules, parameters: nextParams });
                    }}><Plus size={14} /> Add Rule</button>
                  </div>
                  <div className="card">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Distance (Å)</th>
                          <th>Value (Symbol)</th>
                          <th>Ref Pair</th>
                          <th style={{ width: '40px' }}></th>
                        </tr>
                      </thead>
                      <tbody>
                        {config.symmetry_interactions.map((inter, idx) => (
                          <tr key={idx}>
                            <td>
                              <select
                                className="table-select"
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
                            </td>
                            <td><input type="number" step="0.01" className="table-input center" value={inter.distance} onChange={(e) => {
                              const next = [...config.symmetry_interactions]; next[idx].distance = parseFloat(e.target.value); setConfig({ ...config, symmetry_interactions: next })
                            }} /></td>
                            <td><input type="text" className="table-input accent-text" value={inter.value} onChange={(e) => {
                              const next = [...config.symmetry_interactions]; next[idx].value = e.target.value; setConfig({ ...config, symmetry_interactions: next })
                            }} /></td>
                            <td className="text-xs mono opacity-60">
                              {inter.ref_pair ? inter.ref_pair.join('-') : 'Auto'}
                            </td>
                            <td><button onClick={() => {
                              const next = config.symmetry_interactions.filter((_, i) => i !== idx); setConfig({ ...config, symmetry_interactions: next })
                            }} className="icon-btn text-error"><Trash2 size={14} /></button></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-md">
                    <h3 className="section-title text-sm mb-sm">Neighbor Shell Suggestions</h3>
                    <div className="flex gap-2 mb-sm items-center">
                      <button className="btn btn-secondary btn-sm" onClick={fetchNeighbors}>Re-calculate Neighbors</button>
                    </div>
                    {neighborDistances && neighborDistances.length > 0 ? (
                      <div className="code-block" style={{ maxHeight: '200px' }}>
                        {neighborDistances.map((n, i) => (
                          <div key={i} className="flex-between p-xs border-b border-light">
                            <span className="text-xs">
                              d={n.distance.toFixed(3)} Å ({n.count} pairs, ref: {n.ref_pair ? n.ref_pair.join('-') : '?'})
                            </span>
                            <button className="btn btn-primary btn-xs" onClick={() => {
                              // Add rule based on this distance
                              const nextRules = [...config.symmetry_interactions, {
                                type: 'heisenberg',
                                distance: n.distance,
                                value: `J${i + 1}`,
                                ref_pair: n.ref_pair
                              }];
                              const nextParams = { ...config.parameters };
                              if (!nextParams[`J${i + 1}`]) nextParams[`J${i + 1}`] = 0.0;
                              setConfig({ ...config, symmetry_interactions: nextRules, parameters: nextParams });
                            }}>Add J{i + 1}</button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs opacity-60 italic">No neighbor suggestions available. Try recalculating.</p>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex-between mb-md">
                    <h2 className="section-title">Explicit Interactions (Manual)</h2>
                    <button className="btn btn-primary btn-sm" onClick={() => {
                      const next = [...(config.explicit_interactions || [])];
                      next.push({ type: 'heisenberg', distance: 3.0, value: "J1", atom_i: 0, atom_j: 1, offset_j: [0, 0, 0] });
                      setConfig({ ...config, explicit_interactions: next });
                    }}><Plus size={14} /> Add Interaction</button>
                  </div>
                  <div className="card" style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>i</th>
                          <th>j</th>
                          <th>Offset [x,y,z]</th>
                          <th>Value / Vector</th>
                          <th>Dist</th>
                          <th></th>
                        </tr>
                      </thead>
                      <tbody>
                        {(config.explicit_interactions || []).map((inter, idx) => (
                          <tr key={idx}>
                            <td>
                              <select className="table-select" value={inter.type} onChange={(e) => {
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
                            </td>
                            <td><input type="number" className="table-input center w-12" value={inter.atom_i} onChange={(e) => {
                              const next = [...config.explicit_interactions]; next[idx].atom_i = parseInt(e.target.value); setConfig({ ...config, explicit_interactions: next })
                            }} /></td>
                            <td><input type="number" className="table-input center w-12" value={inter.atom_j} onChange={(e) => {
                              const next = [...config.explicit_interactions]; next[idx].atom_j = parseInt(e.target.value); setConfig({ ...config, explicit_interactions: next })
                            }} /></td>
                            <td>
                              <div className="flex gap-1">
                                {[0, 1, 2].map(k => (
                                  <input key={k} type="number" className="table-input center w-10" value={inter.offset_j ? inter.offset_j[k] : 0} onChange={(e) => {
                                    const next = [...config.explicit_interactions];
                                    if (!next[idx].offset_j) next[idx].offset_j = [0, 0, 0];
                                    next[idx].offset_j[k] = parseInt(e.target.value);
                                    setConfig({ ...config, explicit_interactions: next })
                                  }} />
                                ))}
                              </div>
                            </td>
                            <td>
                              {Array.isArray(inter.value) ? (
                                <div className="flex gap-1 flex-col">
                                  <input type="text" className="table-input text-xs" value={inter.value[0]} onChange={(e) => {
                                    const next = [...config.explicit_interactions]; next[idx].value[0] = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                  }} placeholder="Dx" />
                                  <input type="text" className="table-input text-xs" value={inter.value[1]} onChange={(e) => {
                                    const next = [...config.explicit_interactions]; next[idx].value[1] = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                  }} placeholder="Dy" />
                                  <input type="text" className="table-input text-xs" value={inter.value[2]} onChange={(e) => {
                                    const next = [...config.explicit_interactions]; next[idx].value[2] = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                  }} placeholder="Dz" />
                                </div>
                              ) : (
                                <input type="text" className="table-input accent-text" value={inter.value} onChange={(e) => {
                                  const next = [...config.explicit_interactions]; next[idx].value = e.target.value; setConfig({ ...config, explicit_interactions: next })
                                }} />
                              )}
                            </td>
                            <td><input type="number" step="0.01" className="table-input center w-16" value={inter.distance} onChange={(e) => {
                              const next = [...config.explicit_interactions]; next[idx].distance = parseFloat(e.target.value); setConfig({ ...config, explicit_interactions: next })
                            }} /></td>
                            <td><button onClick={() => {
                              const next = config.explicit_interactions.filter((_, i) => i !== idx); setConfig({ ...config, explicit_interactions: next })
                            }} className="icon-btn text-error"><Trash2 size={14} /></button></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'params' && (
            <div className="form-section">
              <h2 className="section-title">Environment Settings</h2>
              <div className="card mb-lg">
                <div className="grid-form">
                  <div className="input-group">
                    <label>Applied Field (T)</label>
                    <input type="number" value={config.parameters.H_mag} onChange={(e) => updateField('parameters', 'H_mag', parseFloat(e.target.value))} />
                  </div>
                  <div className="input-group">
                    <label>Field Direction (h,k,l)</label>
                    <div className="flex-gap-xs">
                      {[0, 1, 2].map(i => (
                        <input key={i} type="number" value={config.parameters.H_dir[i]} onChange={(e) => {
                          const next = [...config.parameters.H_dir]; next[i] = parseFloat(e.target.value); updateField('parameters', 'H_dir', next)
                        }} />
                      ))}
                    </div>
                  </div>
                  <div className="input-group">
                    <label>Default Spin (S)</label>
                    <input type="number" step="0.5" value={config.parameters.S} onChange={(e) => updateField('parameters', 'S', parseFloat(e.target.value))} />
                  </div>
                </div>
              </div>

              <div className="flex-between mb-md">
                <h2 className="section-title">Model Parameters</h2>
                <button className="btn btn-primary btn-sm" onClick={() => {
                  const name = prompt("Enter parameter name (e.g. J1):")
                  if (name) {
                    setConfig(prev => ({
                      ...prev,
                      parameters: { ...prev.parameters, [name]: 0.0 }
                    }))
                  }
                }}><Plus size={14} /> Add Parameter</button>
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
                  <div className="flex-col gap-sm mt-md">
                    {Object.keys(config.tasks).map(taskKey => (
                      <label key={taskKey} className="flex-gap-sm pointer">
                        <input
                          type="checkbox"
                          checked={config.tasks[taskKey]}
                          onChange={(e) => updateField('tasks', taskKey, e.target.checked)}
                        />
                        <span className="text-sm opacity-80">{taskKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="card shadow-glow">
                  <h3>Minimization Parameters</h3>
                  <div className="grid-form mt-md">
                    <div className="input-group">
                      <label>Num Starts</label>
                      <input type="number" value={config.minimization.num_starts}
                        onChange={(e) => updateField('minimization', 'num_starts', parseInt(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>N Workers</label>
                      <input type="number" value={config.minimization.n_workers}
                        onChange={(e) => updateField('minimization', 'n_workers', parseInt(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Early Stopping</label>
                      <input type="number" value={config.minimization.early_stopping}
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
                </div>
              </div>

              <div className="grid-2 mt-md">
                <div className="card shadow-glow">
                  <h3>Display Parameters</h3>
                  <div className="grid-form mt-md">
                    <div className="input-group">
                      <label>Energy Min (meV)</label>
                      <input type="number" step="0.1" value={config.plotting.energy_min}
                        onChange={(e) => updateField('plotting', 'energy_min', parseFloat(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Energy Max (meV)</label>
                      <input type="number" step="0.1" value={config.plotting.energy_max}
                        onChange={(e) => updateField('plotting', 'energy_max', parseFloat(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Broadening (meV)</label>
                      <input type="number" step="0.01" value={config.plotting.broadening}
                        onChange={(e) => updateField('plotting', 'broadening', parseFloat(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Energy Res. (meV)</label>
                      <input type="number" step="0.01" value={config.plotting.energy_resolution}
                        onChange={(e) => updateField('plotting', 'energy_resolution', parseFloat(e.target.value))} />
                    </div>
                    <div className="input-group">
                      <label>Momentum Max (Å⁻¹)</label>
                      <input type="number" step="0.1" value={config.plotting.momentum_max}
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
                  <div className="flex-gap-sm mb-md flex-wrap">
                    {config.q_path.path.map((p, idx) => (
                      <div key={idx} className="badge glass flex-gap-xs align-center">
                        {p}
                        <button className="icon-btn" onClick={() => {
                          const nextPath = config.q_path.path.filter((_, i) => i !== idx);
                          setConfig({ ...config, q_path: { ...config.q_path, path: nextPath } });
                        }}><Trash2 size={10} /></button>
                      </div>
                    ))}
                  </div>
                  <div className="flex-gap-sm align-center">
                    <select className="table-select" id="point-select" style={{ maxWidth: '150px' }}>
                      <option value="">Select Point...</option>
                      {Object.keys(config.q_path.points).map(p => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                    <button className="btn btn-secondary btn-sm" onClick={() => {
                      const sel = document.getElementById('point-select');
                      if (sel.value) {
                        setConfig({ ...config, q_path: { ...config.q_path, path: [...config.q_path.path, sel.value] } });
                        sel.value = "";
                      }
                    }}><Plus size={14} /> Add to Path</button>
                    <div className="ml-auto flex-gap-sm align-center">
                      <span className="text-xxs opacity-60">Points per segment:</span>
                      <input
                        type="number"
                        value={config.q_path.points_per_segment}
                        onChange={(e) => updateField('q_path', 'points_per_segment', parseInt(e.target.value))}
                        className="minimal-input"
                        style={{ width: '60px' }}
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
                <label className="flex-gap-sm pointer glass p-xs px-sm rounded-full border border-light">
                  <input
                    type="checkbox"
                    checked={config.magnetic_structure.enabled}
                    onChange={(e) => updateField('magnetic_structure', 'enabled', e.target.checked)}
                  />
                  <span className="text-sm font-bold vibrant-text">Include Manual Structure</span>
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
                                      <input key={i} type="number" step="0.001" value={dir[i]} onChange={(e) => {
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

        {showVisualizer && (
          <aside className="right-preview glass">
            <div className="preview-container">
              <Visualizer atoms={config.wyckoff_atoms} />
            </div>
          </aside>
        )}
      </main>
      {notification && (
        <div className={`notification ${notification.type}`}>
          {notification.msg}
        </div>
      )}
    </div>
  )
}

export default App
