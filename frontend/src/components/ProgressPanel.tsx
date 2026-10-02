import { useAppStore } from '../store/appStore'

export default function ProgressPanel() {
  const {
    isProgressPanelOpen, toggleProgressPanel,
    graphData, learningPath, progress, selectNode,
  } = useAppStore()

  if (!isProgressPanelOpen || !graphData) return null

  const total = graphData.nodes.length
  const learnedIds = Object.entries(progress)
    .filter(([, s]) => s === 'learned')
    .map(([id]) => id)
  const inProgressIds = Object.entries(progress)
    .filter(([, s]) => s === 'in_progress')
    .map(([id]) => id)

  const percent = total > 0 ? Math.round((learnedIds.length / total) * 100) : 0

  // Suggest next: first unexplored node in learning path
  const nodeById = Object.fromEntries(graphData.nodes.map((n) => [n.id, n]))
  const suggestedNext = learningPath.find(
    (id) => !progress[id] || progress[id] === 'not_started'
  )

  const circumference = 2 * Math.PI * 36  // r=36
  const dashOffset = circumference * (1 - percent / 100)

  return (
    <aside
      className="w-72 shrink-0 flex flex-col border-l border-[#2e3142] bg-[#1a1d27]
                  animate-slide-in overflow-hidden"
      aria-label="Progress panel"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2e3142]">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          ✅ Progress
        </h3>
        <button onClick={toggleProgressPanel}
                className="text-slate-500 hover:text-white text-lg"
                aria-label="Close progress panel">×</button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* Circular progress */}
        <div className="flex flex-col items-center mb-6" aria-label={`${percent}% explored`}>
          <svg width="96" height="96" viewBox="0 0 96 96" aria-hidden="true">
            <circle cx="48" cy="48" r="36" fill="none" stroke="#2e3142" strokeWidth="8" />
            <circle cx="48" cy="48" r="36" fill="none"
                    stroke="#6366f1" strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={dashOffset}
                    strokeLinecap="round"
                    transform="rotate(-90 48 48)"
                    style={{ transition: 'stroke-dashoffset 500ms ease' }} />
            <text x="48" y="48" textAnchor="middle" dominantBaseline="central"
                  fill="white" fontSize="18" fontWeight="600">{percent}%</text>
          </svg>
          <p className="text-slate-400 text-xs mt-1">{learnedIds.length} of {total} concepts</p>
        </div>

        {/* Suggested next */}
        {suggestedNext && (
          <div className="mb-4 p-3 bg-indigo-500/10 border border-indigo-500/30 rounded-lg">
            <p className="text-xs text-indigo-400 font-medium mb-1">Suggested next</p>
            <button onClick={() => selectNode(suggestedNext)}
                    className="text-sm text-white font-medium hover:text-indigo-300 transition
                               text-left w-full">
              {nodeById[suggestedNext]?.label}
            </button>
          </div>
        )}

        {/* In progress */}
        {inProgressIds.length > 0 && (
          <div className="mb-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">In Progress</p>
            <div className="flex flex-wrap gap-1.5">
              {inProgressIds.map((id) => (
                <button key={id} onClick={() => selectNode(id)}
                        className="text-xs bg-amber-500/20 text-amber-400 px-2.5 py-1
                                   rounded-full hover:bg-amber-500/30 transition">
                  {nodeById[id]?.label ?? id.slice(0, 8)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Learned */}
        {learnedIds.length > 0 && (
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Learned</p>
            <div className="flex flex-wrap gap-1.5">
              {learnedIds.map((id) => (
                <button key={id} onClick={() => selectNode(id)}
                        className="text-xs bg-green-500/20 text-green-400 px-2.5 py-1
                                   rounded-full hover:bg-green-500/30 transition">
                  ✓ {nodeById[id]?.label ?? id.slice(0, 8)}
                </button>
              ))}
            </div>
          </div>
        )}

        {learnedIds.length === 0 && inProgressIds.length === 0 && (
          <p className="text-slate-600 text-sm text-center mt-4">
            Click on a concept and mark it as learned or in-progress to track your journey.
          </p>
        )}
      </div>
    </aside>
  )
}
