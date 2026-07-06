import React from 'react'
import { Box, Magnet, Settings, Activity, Wind, BarChart2, Target } from 'lucide-react'

const TABS = [
  { id: 'structure',    label: 'Structure',        icon: Box },
  { id: 'interactions', label: 'Interactions',     icon: Magnet },
  { id: 'params',       label: 'Environment',      icon: Settings },
  { id: 'tasks',        label: 'Tasks & Plotting', icon: Activity },
  { id: 'magstruct',    label: 'Mag. Structure',   icon: Wind },
  { id: 'fitting',      label: 'Data Fitting',     icon: Target },
]

export default function Sidebar({ activeTab, setActiveTab, width }) {
  return (
    <aside className="sidebar glass" style={{ width }}>
      <nav>
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-item ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={20} /> {label}
          </button>
        ))}
        <div className="nav-divider" />
        <button
          className={`nav-item ${activeTab === 'run' ? 'active' : ''}`}
          onClick={() => setActiveTab('run')}
        >
          <BarChart2 size={20} /> Run & Analyze
        </button>
      </nav>
    </aside>
  )
}
