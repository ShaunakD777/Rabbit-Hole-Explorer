import { useState } from 'react'
import { useQuery, useMutation } from 'react-query'
import { useAppStore } from '../store/appStore'
import { getNodeSummary, getNodeSources, expandNode, upsertProgress, extractVideo } from '../api/client'
import type { SummaryDepth, ProgressStatus, SourceDoc, VideoExtractionData, VideoQuizQuestion } from '../types'

const DEPTH_LABELS: Record<SummaryDepth, string> = {
  '2min': '2-min',
  '10min': '10-min',
  'deepdive': 'Deep Dive',
}

const SOURCE_ICONS: Record<string, string> = {
  wikipedia: '📖',
  youtube:   '▶️',
  reddit:    '💬',
  paper:     '📄',
  web:       '🌐',
}

export default function NodeDetailDrawer() {
  const {
    selectedNodeId, isDetailPanelOpen, graphData,
    closeDetailPanel, updateProgress, setExpandingNode, progress,
  } = useAppStore()

  const [tab, setTab] = useState<'summary' | 'sources' | 'related'>('summary')
  const [depth, setDepth] = useState<SummaryDepth>('2min')

  const node = graphData?.nodes.find((n) => n.id === selectedNodeId)

  const { data: summaryData, isLoading: summaryLoading } = useQuery(
    ['summary', selectedNodeId, depth],
    () => getNodeSummary(selectedNodeId!, depth).then((r) => r.data),
    { enabled: !!selectedNodeId && tab === 'summary', staleTime: 1000 * 60 * 30 }
  )

  const { data: sources = [], isLoading: sourcesLoading } = useQuery(
    ['sources', selectedNodeId],
    () => getNodeSources(selectedNodeId!).then((r) => r.data),
    { enabled: !!selectedNodeId && tab === 'sources', staleTime: 1000 * 60 * 10 }
  )

  const expandMutation = useMutation(
    () => expandNode(selectedNodeId!),
    { onSuccess: () => setExpandingNode(selectedNodeId) }
  )

  const progressMutation = useMutation(
    (status: ProgressStatus) => upsertProgress(selectedNodeId!, status),
    {
      onSuccess: (_, status) => updateProgress(selectedNodeId!, status),
    }
  )

  if (!isDetailPanelOpen || !node) return null

  // Related nodes
  const related = graphData?.edges
    .filter((e) => e.source_node_id === node.id || e.target_node_id === node.id)
    .map((e) => {
      const otherId = e.source_node_id === node.id ? e.target_node_id : e.source_node_id
      const other = graphData.nodes.find((n) => n.id === otherId)
      return other ? { node: other, relation: e.relation_type, dir: e.source_node_id === node.id ? '→' : '←' } : null
    })
    .filter(Boolean) ?? []

  const currentStatus = progress[node.id] ?? 'not_started'

  return (
    <aside
      className="w-[360px] shrink-0 flex flex-col border-l border-[#2e3142] bg-[#1a1d27]
                  animate-slide-in overflow-hidden"
      aria-label="Node detail panel"
    >
      {/* Header */}
      <div className="flex items-start justify-between px-4 py-3 border-b border-[#2e3142]">
        <div className="flex-1 min-w-0">
          <h3 className="text-white font-semibold text-base leading-tight truncate">
            {node.label}
          </h3>
          <div className="flex gap-2 mt-1 flex-wrap">
            {node.category && (
              <span className="text-xs text-slate-500 bg-[#2e3142] px-2 py-0.5 rounded-full">
                {node.category}
              </span>
            )}
            {node.is_trending && (
              <span className="text-xs text-orange-400 bg-orange-500/10 px-2 py-0.5 rounded-full">
                🔥 Trending
              </span>
            )}
          </div>
        </div>
        <button onClick={closeDetailPanel}
                className="text-slate-500 hover:text-white ml-2 mt-0.5 text-lg leading-none"
                aria-label="Close panel">
          ×
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[#2e3142]">
        {(['summary', 'sources', 'related'] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
                  className={`flex-1 py-2.5 text-xs font-medium transition uppercase tracking-wide
                    ${tab === t
                      ? 'text-indigo-400 border-b-2 border-indigo-400'
                      : 'text-slate-500 hover:text-slate-300'}`}>
            {t}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-4">
        {tab === 'summary' && (
          <SummaryTab
            depth={depth}
            setDepth={setDepth}
            loading={summaryLoading}
            content={summaryData?.content}
            description={node.description_short}
          />
        )}
        {tab === 'sources' && (
          <SourcesTab sources={sources as SourceDoc[]} loading={sourcesLoading} />
        )}
        {tab === 'related' && (
          <RelatedTab related={related as RelatedItem[]} />
        )}
      </div>

      {/* Footer actions */}
      <div className="px-4 py-3 border-t border-[#2e3142] flex gap-2">
        <button
          onClick={() => progressMutation.mutate('learned')}
          disabled={currentStatus === 'learned'}
          className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-green-600/30
                     text-white text-xs font-medium py-2 rounded-lg transition">
          {currentStatus === 'learned' ? '✓ Learned' : 'Mark Learned'}
        </button>
        <button
          onClick={() => progressMutation.mutate('in_progress')}
          disabled={currentStatus === 'in_progress'}
          className="flex-1 bg-amber-600 hover:bg-amber-700 disabled:bg-amber-600/30
                     text-white text-xs font-medium py-2 rounded-lg transition">
          In Progress
        </button>
        <button
          onClick={() => expandMutation.mutate()}
          disabled={expandMutation.isLoading}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40
                     text-white text-xs font-medium py-2 rounded-lg transition">
          {expandMutation.isLoading ? 'Expanding…' : '+ Expand'}
        </button>
      </div>
    </aside>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SummaryTab({ depth, setDepth, loading, content, description }: {
  depth: SummaryDepth
  setDepth: (d: SummaryDepth) => void
  loading: boolean
  content?: string
  description?: string | null
}) {
  return (
    <div>
      {/* Depth selector */}
      <div className="flex gap-1 mb-4" role="group" aria-label="Summary depth">
        {(Object.keys(DEPTH_LABELS) as SummaryDepth[]).map((d) => (
          <button key={d} onClick={() => setDepth(d)}
                  className={`flex-1 text-xs py-1.5 rounded-lg transition
                    ${depth === d
                      ? 'bg-indigo-500 text-white'
                      : 'bg-[#2e3142] text-slate-400 hover:text-white'}`}>
            {DEPTH_LABELS[d]}
          </button>
        ))}
      </div>
      {loading ? (
        <div className="space-y-2 animate-pulse">
          {[...Array(5)].map((_, i) => (
            <div key={i} className={`h-3 bg-[#2e3142] rounded ${i === 4 ? 'w-3/4' : 'w-full'}`} />
          ))}
        </div>
      ) : content ? (
        <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
      ) : description ? (
        <p className="text-slate-400 text-sm leading-relaxed">{description}</p>
      ) : (
        <p className="text-slate-600 text-sm">No summary available yet.</p>
      )}
    </div>
  )
}

function SourcesTab({ sources, loading }: { sources: SourceDoc[]; loading: boolean }) {
  if (loading) return <SkeletonList />
  if (!sources.length) return <p className="text-slate-600 text-sm">No sources found.</p>
  return (
    <ul className="space-y-3">
      {sources.map((s) => (
        <li key={s.id} className="bg-[#0f1117] rounded-lg p-3 border border-[#2e3142]">
          <a href={s.url} target="_blank" rel="noopener noreferrer"
             className="flex items-start gap-2 group">
            <span className="text-lg shrink-0">{SOURCE_ICONS[s.source_type] ?? '🔗'}</span>
            <div className="min-w-0">
              <p className="text-sm text-slate-200 group-hover:text-white leading-tight
                            line-clamp-2 transition">
                {s.title}
              </p>
              {s.author_or_channel && (
                <p className="text-xs text-slate-500 mt-0.5 truncate">{s.author_or_channel}</p>
              )}
            </div>
          </a>
          {/* POST /videos/{id}/extract has existed since the backend was written, but no
              frontend component ever called it -- wire it up for YouTube sources. */}
          {s.source_type === 'youtube' && <VideoNotesSection sourceDocId={s.id} />}
        </li>
      ))}
    </ul>
  )
}

function VideoNotesSection({ sourceDocId }: { sourceDocId: string }) {
  const [open, setOpen] = useState(false)
  const mutation = useMutation(() =>
    extractVideo(sourceDocId).then((r) => r.data as VideoExtractionData)
  )

  if (!open) {
    return (
      <button
        onClick={() => { setOpen(true); mutation.mutate() }}
        className="mt-2 pt-2 border-t border-[#2e3142] w-full text-left text-xs
                   text-indigo-400 hover:text-indigo-300 font-medium">
        🎬 Generate notes &amp; quiz
      </button>
    )
  }

  return (
    <div className="mt-2 pt-2 border-t border-[#2e3142] space-y-2">
      {mutation.isLoading && <p className="text-xs text-slate-500">Generating notes…</p>}
      {mutation.isError && (
        <p className="text-xs text-red-400">
          Couldn't generate notes (no transcript, or no LLM provider available).
        </p>
      )}
      {mutation.data && (
        <>
          <p className="text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
            {mutation.data.notes_outline}
          </p>
          {mutation.data.key_concepts.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {mutation.data.key_concepts.map((c, i) => (
                <span key={i}
                      className="text-[10px] bg-[#2e3142] text-slate-400 px-2 py-0.5 rounded-full">
                  {c}
                </span>
              ))}
            </div>
          )}
          {mutation.data.quiz.map((q, i) => (
            <QuizQuestion key={i} question={q} />
          ))}
        </>
      )}
    </div>
  )
}

function QuizQuestion({ question }: { question: VideoQuizQuestion }) {
  const [revealed, setRevealed] = useState(false)
  return (
    <div className="bg-[#1a1d27] rounded-lg p-2">
      <p className="text-xs text-slate-300 mb-1.5">{question.question}</p>
      <div className="flex flex-col gap-1">
        {question.options.map((opt, i) => (
          <button key={i} onClick={() => setRevealed(true)}
                  className={`text-left text-xs px-2 py-1 rounded transition
                    ${revealed && i === question.correct_index
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-[#2e3142] text-slate-400 hover:text-white'}`}>
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}

type RelatedItem = {
  node: { id: string; label: string; category: string | null }
  relation: string
  dir: string
}

function RelatedTab({ related }: { related: RelatedItem[] }) {
  const selectNode = useAppStore((s) => s.selectNode)
  if (!related.length) return <p className="text-slate-600 text-sm">No related concepts.</p>
  return (
    <ul className="space-y-2">
      {related.map(({ node, relation, dir }) => (
        <li key={node.id}>
          <button onClick={() => selectNode(node.id)}
                  className="w-full text-left flex items-center gap-3 p-2.5 rounded-lg
                             bg-[#0f1117] border border-[#2e3142] hover:border-indigo-500
                             transition group">
            <span className="text-indigo-400 text-xs font-mono shrink-0">{dir}</span>
            <div className="min-w-0">
              <p className="text-sm text-slate-200 group-hover:text-white truncate">{node.label}</p>
              <p className="text-xs text-slate-500">{relation.replace(/_/g, ' ')}</p>
            </div>
          </button>
        </li>
      ))}
    </ul>
  )
}

function SkeletonList() {
  return (
    <ul className="space-y-3 animate-pulse">
      {[...Array(4)].map((_, i) => (
        <li key={i} className="h-14 bg-[#2e3142] rounded-lg" />
      ))}
    </ul>
  )
}
