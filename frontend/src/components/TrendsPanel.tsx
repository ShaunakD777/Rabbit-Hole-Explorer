import { useQuery } from 'react-query'
import { useAppStore } from '../store/appStore'
import { getTrends } from '../api/client'
import type { TrendData } from '../types'

export default function TrendsPanel() {
  const { topicId, isTrendsPanelOpen, toggleTrendsPanel } = useAppStore()

  const { data: trends = [], isLoading } = useQuery(
    ['trends', topicId],
    () => getTrends(topicId!).then((r) => r.data as TrendData[]),
    { enabled: !!topicId && isTrendsPanelOpen, staleTime: 1000 * 60 * 30 }
  )

  if (!isTrendsPanelOpen) return null

  return (
    <aside
      className="w-72 shrink-0 flex flex-col border-l border-[#2e3142] bg-[#1a1d27]
                  animate-slide-in overflow-hidden"
      aria-label="Research trends panel"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2e3142]">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          📈 Research Trends
        </h3>
        <button onClick={toggleTrendsPanel}
                className="text-slate-500 hover:text-white text-lg"
                aria-label="Close trends panel">×</button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {isLoading && (
          <div className="space-y-2 animate-pulse">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-16 bg-[#2e3142] rounded-lg" />
            ))}
          </div>
        )}

        {!isLoading && trends.length === 0 && (
          <p className="text-slate-600 text-sm text-center mt-8">
            No trend data available for this topic yet.
          </p>
        )}

        {!isLoading && trends.map((t) => {
          // growth_rate from the backend is a ratio (e.g. 1.0 == +100%), not a
          // percentage -- scale it here so "100% growth" doesn't render as "+1%".
          const growthPct = t.growth_rate != null ? t.growth_rate * 100 : null
          return (
          <div key={t.id}
               className="bg-[#0f1117] border border-[#2e3142] rounded-lg p-3 mb-2">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm text-slate-200 leading-tight">{t.cluster_label}</p>
              {growthPct != null && (
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full shrink-0
                  ${growthPct > 50
                    ? 'bg-orange-500/20 text-orange-400'
                    : 'bg-green-500/20 text-green-400'}`}>
                  {growthPct > 0 ? '+' : ''}{growthPct.toFixed(0)}%
                </span>
              )}
            </div>
            {t.window_start && (
              <p className="text-xs text-slate-600 mt-1">
                {t.window_start} → {t.window_end}
              </p>
            )}
          </div>
          )
        })}
      </div>
    </aside>
  )
}
