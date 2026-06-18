import { useState, useEffect, useCallback } from 'react'
import { Route, Switch, Link, useLocation } from 'wouter'

// ── API client ──────────────────────────────────────────────────────
async function api(path: string, opts?: RequestInit) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  return res.json()
}

// ── Types ───────────────────────────────────────────────────────────
interface TableInfo { table: string; count: number }
interface Record { id: string; table: string; data: any; created_at: string; updated_at: string }
interface SearchResult extends Record { score: number }

// ── Dashboard ───────────────────────────────────────────────────────
function Dashboard() {
  const [tables, setTables] = useState<TableInfo[]>([])
  const [info, setInfo] = useState<any>(null)
  const [health, setHealth] = useState<string>('...')

  useEffect(() => {
    api('/tables').then(setTables).catch(() => {})
    api('/info').then(setInfo).catch(() => {})
    api('/health').then((d) => setHealth(d.status)).catch(() => setHealth('error'))
  }, [])

  const totalRecords = tables.reduce((s, t) => s + t.count, 0)

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
      <div className="grid grid-cols-3 gap-4 mb-6">
        <StatCard label="Status" value={health === 'ok' ? 'Healthy' : health} />
        <StatCard label="Tables" value={String(tables.length)} />
        <StatCard label="Records" value={String(totalRecords)} />
      </div>
      {info && (
        <div className="bg-gray-800 rounded p-4 mb-4">
          <h3 className="font-semibold mb-2">Database</h3>
          <p className="text-sm text-gray-400">{info.path}</p>
          <p className="text-sm text-gray-400">Size: {formatBytes(info.total_size)}</p>
          <p className="text-sm text-gray-400">Plugins: {Object.keys(info.plugins || {}).join(', ') || 'none'}</p>
        </div>
      )}
      <h3 className="font-semibold mb-2">Tables</h3>
      <div className="grid grid-cols-2 gap-2">
        {tables.map((t) => (
          <Link key={t.table} href={`/records?table=${t.table}`}>
            <div className="bg-gray-800 rounded p-3 hover:bg-gray-700 cursor-pointer">
              <span className="font-mono text-sm">{t.table}</span>
              <span className="text-gray-400 text-sm ml-2">({t.count})</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800 rounded p-4">
      <div className="text-gray-400 text-sm">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  )
}

// ── Records Browser ─────────────────────────────────────────────────
function RecordsBrowser() {
  const [records, setRecords] = useState<Record[]>([])
  const [tables, setTables] = useState<TableInfo[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [selected, setSelected] = useState<Record | null>(null)

  useEffect(() => { api('/tables').then(setTables).catch(() => {}) }, [])

  const loadRecords = useCallback((table: string) => {
    setSelectedTable(table)
    const q = table ? `?table=${table}&limit=100` : '?limit=100'
    api(`/records${q}`).then(setRecords).catch(() => {})
  }, [])

  // Load from URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const t = params.get('table')
    if (t) loadRecords(t)
  }, [loadRecords])

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Records</h2>
      <div className="flex gap-2 mb-4 flex-wrap">
        {tables.map((t) => (
          <button
            key={t.table}
            onClick={() => loadRecords(t.table)}
            className={`px-3 py-1 rounded text-sm ${selectedTable === t.table ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
          >{t.table} ({t.count})</button>
        ))}
      </div>
      <div className="flex gap-4">
        <div className="flex-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-700">
                <th className="py-2">ID</th>
                <th>Table</th>
                <th>Created</th>
                <th>Preview</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.id} onClick={() => setSelected(r)}
                  className="border-b border-gray-800 hover:bg-gray-800 cursor-pointer">
                  <td className="py-1 font-mono text-xs">{r.id.slice(0, 12)}...</td>
                  <td>{r.table}</td>
                  <td className="text-gray-400 text-xs">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="text-xs text-gray-400 truncate max-w-xs">{JSON.stringify(r.data).slice(0, 60)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {selected && (
          <div className="w-96 bg-gray-800 rounded p-4">
            <div className="flex justify-between mb-2">
              <h3 className="font-semibold">Record Detail</h3>
              <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-white">x</button>
            </div>
            <p className="text-xs font-mono mb-1">ID: {selected.id}</p>
            <p className="text-xs text-gray-400 mb-2">Table: {selected.table}</p>
            <pre className="text-xs bg-gray-900 p-2 rounded overflow-auto max-h-96">
              {JSON.stringify(selected.data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Query Console ───────────────────────────────────────────────────
function QueryConsole() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [mode, setMode] = useState<'vector' | 'fts' | 'recall'>('recall')
  const [loading, setLoading] = useState(false)

  const runQuery = async () => {
    setLoading(true)
    try {
      const endpoint = mode === 'fts' ? '/fts' : mode === 'vector' ? '/search' : '/recall'
      const data = await api(endpoint, {
        method: 'POST',
        body: JSON.stringify({ query, top_k: 10, limit: 10 }),
      })
      setResults(Array.isArray(data) ? data : [])
    } catch { setResults([]) }
    setLoading(false)
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Query Console</h2>
      <div className="flex gap-2 mb-4">
        {(['recall', 'vector', 'fts'] as const).map((m) => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-3 py-1 rounded text-sm ${mode === m ? 'bg-blue-600' : 'bg-gray-700'}`}
          >{m.toUpperCase()}</button>
        ))}
      </div>
      <div className="flex gap-2 mb-4">
        <input
          value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && runQuery()}
          placeholder="Enter search query..."
          className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
        />
        <button onClick={runQuery} disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded text-sm disabled:opacity-50"
        >{loading ? '...' : 'Search'}</button>
      </div>
      {results.length > 0 && (
        <div className="space-y-2">
          {results.map((r, i) => (
            <div key={r.id || i} className="bg-gray-800 rounded p-3">
              <div className="flex justify-between text-sm">
                <span className="font-mono text-xs">{r.id?.slice(0, 16)}</span>
                <span className="text-gray-400">{r.table}</span>
                {r.score !== undefined && (
                  <span className="text-blue-400">score: {r.score.toFixed(3)}</span>
                )}
              </div>
              <pre className="text-xs mt-1 text-gray-300 overflow-auto max-h-32">
                {JSON.stringify(r.data, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
      {results.length === 0 && query && !loading && (
        <p className="text-gray-500 text-sm">No results</p>
      )}
    </div>
  )
}

// ── Health ───────────────────────────────────────────────────────────
function HealthPage() {
  const [report, setReport] = useState<any>(null)
  const [dbStats, setDbStats] = useState<any>(null)

  useEffect(() => {
    api('/doctor').then(setReport).catch(() => {})
    api('/stats').then(setDbStats).catch(() => {})
  }, [])

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Health & Stats</h2>
      {report && (
        <div className="bg-gray-800 rounded p-4 mb-4">
          <h3 className="font-semibold mb-2">Doctor Report</h3>
          <pre className="text-xs overflow-auto max-h-96">
            {JSON.stringify(report, null, 2)}
          </pre>
        </div>
      )}
      {dbStats && (
        <div className="bg-gray-800 rounded p-4">
          <h3 className="font-semibold mb-2">Database Stats</h3>
          <pre className="text-xs overflow-auto max-h-96">
            {JSON.stringify(dbStats, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ── Sidebar ─────────────────────────────────────────────────────────
function Sidebar() {
  const [location] = useLocation()
  const links = [
    { path: '/', label: 'Dashboard', icon: '~' },
    { path: '/records', label: 'Records', icon: '#' },
    { path: '/query', label: 'Query', icon: '>' },
    { path: '/health', label: 'Health', icon: '+' },
  ]

  return (
    <nav className="w-48 bg-gray-900 border-r border-gray-800 p-4 flex flex-col gap-1">
      <h1 className="text-lg font-bold mb-4 text-blue-400">Axil</h1>
      {links.map((l) => (
        <Link key={l.path} href={l.path}>
          <div className={`px-3 py-2 rounded text-sm cursor-pointer ${location === l.path ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'}`}>
            <span className="mr-2 font-mono">{l.icon}</span>{l.label}
          </div>
        </Link>
      ))}
    </nav>
  )
}

// ── App ─────────────────────────────────────────────────────────────
export default function App() {
  return (
    <div className="flex h-screen bg-gray-950 text-white">
      <Sidebar />
      <main className="flex-1 p-6 overflow-auto">
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/records" component={RecordsBrowser} />
          <Route path="/query" component={QueryConsole} />
          <Route path="/health" component={HealthPage} />
          <Route>
            <div className="text-gray-500">Page not found</div>
          </Route>
        </Switch>
      </main>
    </div>
  )
}

// ── Utils ───────────────────────────────────────────────────────────
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
