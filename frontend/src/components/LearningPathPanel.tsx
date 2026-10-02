import { useAppStore } from '../store/appStore'

export default function LearningPathPanel() {
  const { graphData, learningPath, selectedNodeId, progress, selectNode } = useAppStore()

  if (!graphData) return null

  const nodeById = Object.fromEntries(graphData.nodes.map((n) => [n.id, n]))

  return (
    <aside
      className="w-60 shrink-0 flex flex-col border-r border-[#2e3142] bg-[#1a1d27]
                  overflow-y-auto"
      aria-label="Learning path"
    >
      <div className="px-4 py-3 border-b border-[#2e3142]">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Start Here
        </h3>
      </div>

      <ol className="flex-1 py-2">
        {learningPath.map((nodeId, idx) => {
          const node = nodeById[nodeId]
          if (!node) return null
          const status = progress[nodeId] ?? 'not_started'
          const isSelected = selectedNodeId === nodeId

          return (
            <li key={nodeId}>
              <button
                onClick={() => selectNode(nodeId)}
                className={`w-full text-left flex items-start gap-3 px-4 py-2.5
                  text-sm transition group
                  ${isSelected
                    ? 'bg-indigo-500/20 text-white'
                    : 'text-slate-400 hover:bg-[#2a2d3e] hover:text-white'}`}
                aria-current={isSelected ? 'true' : undefined}
              >
                {/* Step number / status icon */}
                <span className={`shrink-0 w-5 h-5 rounded-full flex items-center
                  justify-center text-xs font-bold mt-0.5
                  ${status === 'learned'
                    ? 'bg-green-500 text-white'
                    : status === 'in_progress'
                    ? 'bg-amber-500 text-white'
                    : isSelected
                    ? 'bg-indigo-500 text-white'
                    : 'bg-[#2e3142] text-slate-400 group-hover:bg-[#3e4257]'}`}
                  aria-label={status}>
                  {status === 'learned' ? '✓' : idx + 1}
                </span>

                <span className="flex-1 leading-tight truncate">{node.label}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </aside>
  )
}
