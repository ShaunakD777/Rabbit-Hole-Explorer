import { useState, FormEvent } from 'react'
import { useAppStore } from '../store/appStore'
import { searchGraph } from '../api/client'

type SearchResult = { node_id: string; label: string; score: number }

export default function TopBar() {
  const {
    topicLabel, graphId, reset,
    toggleTrendsPanel, toggleProgressPanel,
    selectNode,
  } = useAppStore()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)

  const onSearch = async (e: FormEvent) => {
    e.preventDefault()
    if (!graphId || !query.trim()) return
    setSearching(true)
    try {
      const res = await searchGraph(graphId, query)
      setResults(res.data)
    } finally {
      setSearching(false)
    }
  }

  return (
    <header className="h-14 flex items-center gap-4 px-4 border-b border-[#2e3142]
                        bg-[#1a1d27] z-10 shrink-0">
      {/* Topic title */}
      <h2 className="text-white font-semibold text-base truncate max-w-xs" title={topicLabel}>
        🐇 {topicLabel}
      </h2>

      {/* Semantic search */}
      <form onSubmit={onSearch} className="relative flex-1 max-w-md">
        <input
          type="search"
          value={query}
          onChange={(e) => { setQuery(e.target.value); if (!e.target.value) setResults([]) }}
          placeholder={searching ? 'Searching…' : 'Search concepts… (press /)'}
          disabled={searching}
          className="w-full bg-[#0f1117] border border-[#2e3142] rounded-lg px-4 py-2
                     text-sm text-white placeholder-slate-500 outline-none disabled:opacity-60
                     focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30"
          onKeyDown={(e) => e.key === 'Escape' && setResults([])}
          aria-label="Search within graph"
        />
        {results.length > 0 && (
          <ul className="absolute top-full left-0 right-0 mt-1 bg-[#1a1d27] border
                         border-[#2e3142] rounded-lg shadow-xl z-50 max-h-56 overflow-y-auto"
              role="listbox">
            {results.map((r) => (
              <li key={r.node_id}>
                <button
                  onClick={() => { selectNode(r.node_id); setResults([]) }}
                  className="w-full text-left px-4 py-2 text-sm text-slate-300
                             hover:bg-[#2a2d3e] hover:text-white flex justify-between"
                  role="option"
                >
                  <span>{r.label}</span>
                  <span className="text-slate-500 text-xs">{(r.score * 100).toFixed(0)}%</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </form>

      <div className="flex gap-2 ml-auto">
        <button onClick={toggleTrendsPanel}
                className="text-xs text-slate-400 hover:text-white px-3 py-1.5
                           rounded-lg border border-[#2e3142] hover:border-indigo-500 transition">
          📈 Trends
        </button>
        <button onClick={toggleProgressPanel}
                className="text-xs text-slate-400 hover:text-white px-3 py-1.5
                           rounded-lg border border-[#2e3142] hover:border-indigo-500 transition">
          ✅ Progress
        </button>
        <button onClick={reset}
                className="text-xs text-slate-400 hover:text-white px-3 py-1.5
                           rounded-lg border border-[#2e3142] hover:border-red-500 transition">
          ＋ New Topic
        </button>
      </div>
    </header>
  )
}


