import React from 'react'
import { FileText } from 'lucide-react'

export default function LogConsole({ logs, connected }) {
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
          <div className="flex align-center gap-xs ml-sm opacity-100">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-success animate-pulse' : 'bg-error'}`} />
            <span style={{ fontSize: '10px' }}>{connected ? 'CONNECTED' : 'DISCONNECTED'}</span>
          </div>
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
