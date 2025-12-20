import React, { useState } from 'react'
import { Beaker, Database, Activity, Code, Download, Plus, Trash2, Settings, Box, Eye, Share2, Info, Magnet, Wind } from 'lucide-react'
import yaml from 'js-yaml'
import Visualizer from './components/Visualizer'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('structure')
  const [showVisualizer, setShowVisualizer] = useState(true)
  const [config, setConfig] = useState({
    lattice: { a: 7.33, b: 7.33, c: 17.1374, alpha: 90, beta: 90, gamma: 120, space_group: 163 },
    wyckoff_atoms: [
      { label: 'Fe', pos: [0.5, 0, 0], spin_S: 2.5 }
    ],
    symmetry_interactions: [
      { type: 'heisenberg', ref_pair: ['Fe0', 'Fe1'], distance: 3.665, value: 'J1' },
      { type: 'dm', ref_pair: ['Fe1', 'Fe2'], distance: 3.665, value: ['0', '-Dy', '-Dz'] }
    ],
    parameters: { S: 2.5, H_mag: 0.0, H_dir: [0, 0, 1] },
    tasks: { run_minimization: true, run_dispersion: true, plot_dispersion: true }
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
      plotting: {
        save_plot: true,
        show_plot: false
      }
    }

    try {
      const response = await fetch('/api/expand-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: input }),
      })

      if (!response.ok) throw new Error('Failed to expand config')

      const expanded = await response.json()
      const data = yaml.dump(expanded)
      const blob = new Blob([data], { type: 'text/yaml' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'config_expanded.yaml'
      link.click()
    } catch (err) {
      alert('Error expanding config: ' + err.message)
    }
  }

  const handleSaveToDisk = async () => {
    try {
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
        minimization: {
          enabled: true,
          method: "L-BFGS-B",
          maxiter: 3000
        }
      }

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
        body: JSON.stringify({ filename: 'config_pure.yaml', data: expanded }),
      })

      if (!response.ok) throw new Error('Failed to save to disk')
      alert('Saved to config_pure.yaml (expanded) successfully!')
    } catch (err) {
      alert('Error saving: ' + err.message)
    }
  }

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
            <span className="version-tag">STABLE v2.0</span>
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
                  const next = [...config.symmetry_interactions];
                  next.push({ type: 'heisenberg', distance: 3.0, value: 'J1' });
                  setConfig({ ...config, symmetry_interactions: next });
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
            </div>
          )}

          {activeTab === 'params' && (
            <div className="form-section">
              <h2 className="section-title">Environment Settings</h2>
              <div className="card">
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
    </div>
  )
}

export default App
