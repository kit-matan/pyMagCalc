import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, Share2, Info, Magnet, Wind } from 'lucide-react'
import yaml from 'js-yaml'
import Visualizer from './components/Visualizer'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('structure')
  const [showVisualizer, setShowVisualizer] = useState(true)
  const [designerFilename, setDesignerFilename] = useState('config_designer.yaml')
  const [notification, setNotification] = useState(null)
  const [neighborDistances, setNeighborDistances] = useState([])

  const showNotify = (msg, type = 'success') => {
    console.log(`[Notification] ${type}: ${msg}`)
    setNotification({ msg, type })
    setTimeout(() => setNotification(null), 5000)
  }

  const [config, setConfig] = useState({
    lattice: { a: 7.33, b: 7.33, c: 17.1374, alpha: 90, beta: 90, gamma: 120, space_group: 163 },
    wyckoff_atoms: [
      { label: 'Fe', pos: [0.5, 0, 0], spin_S: 2.5 }
    ],
    symmetry_interactions: [
      { type: 'heisenberg', ref_pair: ['Fe0', 'Fe1'], distance: 3.665, value: 'J1' },
      { type: 'dm', ref_pair: ['Fe1', 'Fe2'], distance: 3.665, value: ['0', '-Dy', '-Dz'] }
    ],
    parameters: { S: 2.5, H_mag: 0.0, H_dir: [0, 0, 1], J1: -1.0, Dy: 0.1, Dz: -0.1 },
    tasks: {
      run_minimization: true,
      run_dispersion: true,
      plot_dispersion: true,
      run_sqw_map: false,
      plot_sqw_map: false
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

  const exportYaml = async () => {
    // Structure the input for the Expansion Backend
    const input = {
      crystal_structure: {
        lattice_parameters: config.lattice,
        wyckoff_atoms: config.wyckoff_atoms
      },
      interactions: {
        symmetry_rules: config.symmetry_interactions
      },
      parameters: config.parameters,
      minimization: {
        enabled: true,
        method: "L-BFGS-B",
        maxiter: 3000
      },
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
      console.log('Expansion successful, generating file...')
      const data = yaml.dump(expanded)
      const blob = new Blob([data], { type: 'text/yaml' })
      const url = URL.createObjectURL(blob)

      const link = document.createElement('a')
      link.href = url
      link.download = designerFilename.endsWith('.yaml') ? designerFilename : `${designerFilename}.yaml`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      URL.revokeObjectURL(url)
      showNotify('Configuration exported successfully.')
    } catch (err) {
      console.error('Export error:', err)
      showNotify('Export failed: ' + err.message, 'error')
    }
  }

  const handleSaveToDisk = async () => {
    const input = {
      crystal_structure: {
        lattice_parameters: config.lattice,
        wyckoff_atoms: config.wyckoff_atoms
      },
      interactions: {
        symmetry_rules: config.symmetry_interactions
      },
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
        method: "L-BFGS-B",
        maxiter: 3000
      }
    }

    try {
      console.log('Sending save request to backend...')
      const expandResponse = await fetch('/api/expand-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: input }),
      })

      if (!expandResponse.ok) throw new Error('Failed to expand config before saving')
      const expanded = await expandResponse.json()

      const response = await fetch('/api/save-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: designerFilename, data: expanded }),
      })

      if (!response.ok) throw new Error('Failed to save to disk')
      const result = await response.json()
      showNotify(`Success! Saved to ${result.path}`)
    } catch (err) {
      console.error('Save error:', err)
      showNotify('Error saving: ' + err.message, 'error')
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
              wyckoff_atoms: config.wyckoff_atoms
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
              <div className="filename-input-wrapper glass">
                <Code size={12} className="opacity-60" />
                <input
                  type="text"
                  value={designerFilename}
                  onChange={(e) => setDesignerFilename(e.target.value)}
                  placeholder="filename.yaml"
                  className="filename-input"
                />
              </div>
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
          <button className="btn btn-secondary glass" onClick={exportYaml}>
            <Download size={16} /> Export
          </button>
          <button className="btn btn-primary shadow-glow" onClick={handleSaveToDisk}>
            <Database size={16} /> Save to Disk
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
                        <td><button onClick={() => {
                          const next = config.symmetry_interactions.filter((_, i) => i !== idx); setConfig({ ...config, symmetry_interactions: next })
                        }} className="icon-btn text-error"><Trash2 size={14} /></button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {neighborDistances.length > 0 && (
                <div className="mt-xl">
                  <h3 className="section-subtitle mb-lg">Neighbor Shell Suggestions</h3>
                  {(() => {
                    const groups = {};
                    neighborDistances.forEach(n => {
                      if (!groups[n.shell_label]) groups[n.shell_label] = [];
                      groups[n.shell_label].push(n);
                    });

                    return Object.keys(groups).map(label => (
                      <div key={label} className="mb-lg">
                        <h4 className="text-xs uppercase tracking-widest opacity-40 font-black mb-sm flex items-center gap-xs">
                          <span className="w-8 h-px bg-current opacity-20" /> {label} Neighbors
                        </h4>
                        <div className="neighbor-grid">
                          {groups[label].map((n, i) => (
                            <div key={i} className="neighbor-card glass pointer" onClick={() => {
                              const nextRules = [...config.symmetry_interactions, {
                                type: 'heisenberg',
                                distance: n.distance,
                                value: `J${config.symmetry_interactions.length + 1}`,
                                ref_pair: n.ref_pair
                              }];
                              const nextParams = { ...config.parameters };
                              const pName = `J${config.symmetry_interactions.length + 1}`;
                              if (!nextParams[pName]) nextParams[pName] = 0.0;
                              setConfig({ ...config, symmetry_interactions: nextRules, parameters: nextParams });
                              showNotify(`Added ${n.shell_label} ${n.ref_pair.join('-')} bond at ${n.distance} Å`);
                            }}>
                              <div className="flex-between mb-xs">
                                <span className="mono text-accent">{n.ref_pair.join(' - ')}</span>
                                <div style={{ textAlign: 'right' }}>
                                  <div className="badge">{n.distance} Å</div>
                                  <div className="text-xxs opacity-40 mt-xs" style={{ fontSize: '0.6rem' }}>{n.multiplicity} equivalent bonds</div>
                                </div>
                              </div>
                              <div className="text-xxs opacity-60">Offset: [{n.offset.join(', ')}]</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ));
                  })()}
                </div>
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
