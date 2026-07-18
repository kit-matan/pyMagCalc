import React from 'react'
import { Share2, Magnet, Code, Trash2, Download, FolderOpen, Save } from 'lucide-react'

export default function AppHeader({ onCifUpload, onMcifUpload, onYamlImport, onReset, onExportYaml, onOpenFile, onSave, currentFilePath }) {
  return (
    <header className="glass">
      <div className="logo animate-fade-in">
        <div className="icon-wrapper gradient-bg">
          <img src="/spin_vector_icon.png" alt="Spin Vector Icon" className="w-full h-full object-cover" />
        </div>
        <div>
          <h1 className="header-title">pyMagCalc Studio</h1>
          <div className="flex-gap-xs align-center">
            <span className="subtitle">
              {currentFilePath
                ? `Editing ${currentFilePath}`
                : 'Configure Models & Calculate Spin-Waves'}
            </span>
          </div>
        </div>
      </div>
      <div className="header-actions">
        <button className="btn btn-secondary glass cursor-pointer" onClick={onOpenFile}
          title="Open a config file from disk (the same file magcalc run reads)">
          <FolderOpen size={16} /> Open File
        </button>
        <button className="btn btn-primary shadow-glow" onClick={() => onSave(false)}
          title={currentFilePath ? `Save to ${currentFilePath}` : 'Save to a file on disk'}>
          <Save size={16} /> Save
        </button>
        <label className="btn btn-secondary glass cursor-pointer">
          <Share2 size={16} /> Load CIF
          <input type="file" accept=".cif" hidden onChange={onCifUpload} />
        </label>
        <label className="btn btn-secondary glass cursor-pointer" title="Load a magnetic CIF: expands the magnetic space group into the full magnetic cell (atoms + spin directions)">
          <Magnet size={16} /> Load mCIF
          <input type="file" accept=".mcif,.cif" hidden onChange={onMcifUpload} />
        </label>
        <label className="btn btn-secondary glass cursor-pointer" title="Import a YAML config from the browser (no disk path; Save will prompt for one)">
          <Code size={16} /> Load YAML
          <input type="file" accept=".yaml,.yml" hidden onChange={onYamlImport} />
        </label>
        <button className="btn btn-secondary glass cursor-pointer" onClick={onReset}>
          <Trash2 size={16} /> Load Defaults
        </button>
        <button className="btn btn-secondary glass cursor-pointer" onClick={onExportYaml}
          title="Download the config as a YAML file (browser download)">
          <Download size={16} /> Export YAML
        </button>
      </div>
    </header>
  )
}
