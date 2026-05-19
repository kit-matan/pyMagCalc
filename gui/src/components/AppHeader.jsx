import React from 'react'
import { Share2, Code, Trash2, Download } from 'lucide-react'

export default function AppHeader({ onCifUpload, onYamlImport, onReset, onExportYaml }) {
  return (
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
          <input type="file" accept=".cif" hidden onChange={onCifUpload} />
        </label>
        <label className="btn btn-secondary glass cursor-pointer">
          <Code size={16} /> Load YAML
          <input type="file" accept=".yaml,.yml" hidden onChange={onYamlImport} />
        </label>
        <button className="btn btn-secondary glass cursor-pointer" onClick={onReset}>
          <Trash2 size={16} /> Load Defaults
        </button>
        <button className="btn btn-primary shadow-glow" onClick={onExportYaml}>
          <Download size={16} /> Export YAML
        </button>
      </div>
    </header>
  )
}
