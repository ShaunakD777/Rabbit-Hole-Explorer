import { useState, FormEvent } from 'react'
import { createTopic, clarifyTopic } from '../api/client'
import { useAppStore } from '../store/appStore'
import type { ClarificationOption } from '../types'

const EXAMPLE_TOPICS = [
  'Artificial General Intelligence',
  'Quantum Computing',
  'Blockchain',
  'Large Language Models',
  'Reinforcement Learning',
  'CRISPR Gene Editing',
]

export default function LandingPage() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [clarification, setClarification] = useState<
    { topicId: string; options: ClarificationOption[] } | null
  >(null)
  const setTopic = useAppStore((s) => s.setTopic)

  const submit = async (topic: string) => {
    const trimmed = topic.trim()
    if (!trimmed) return
    setLoading(true)
    setError('')
    setClarification(null)
    try {
      const res = await createTopic(trimmed)
      if (res.data.status === 'needs_clarification') {
        setClarification({ topicId: res.data.id, options: res.data.clarification_options })
        setLoading(false)
        return
      }
      setTopic(res.data.id, res.data.normalized_label)
    } catch (err: unknown) {
      setError('Failed to start exploration. Is the backend running?')
      setLoading(false)
    }
  }

  const chooseClarification = async (option: ClarificationOption) => {
    if (!clarification) return
    setLoading(true)
    setError('')
    try {
      const res = await clarifyTopic(clarification.topicId, option.clarifying_query)
      setClarification(null)
      setTopic(res.data.id, res.data.normalized_label)
    } catch (err: unknown) {
      setError('Failed to start exploration. Is the backend running?')
      setLoading(false)
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    submit(query)
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4"
         style={{ background: 'var(--bg-canvas)' }}>
      {/* Header */}
      <div className="text-center mb-10 animate-fade-in">
        <h1 className="text-4xl font-bold text-white mb-3 tracking-tight">
          🐇 Rabbit Hole Explorer
        </h1>
        <p className="text-slate-400 text-lg max-w-xl mx-auto">
          Type any topic and get an interactive knowledge graph — concepts,
          connections, sources, and a learning path to guide you through the rabbit hole.
        </p>
      </div>

      {clarification ? (
        /* Disambiguation picker: the topic has multiple distinct, unrelated
           common meanings -- ask which one rather than letting whichever
           source happened to load first silently decide. */
        <div className="w-full max-w-2xl animate-fade-in">
          <p className="text-slate-300 text-center mb-5">
            Which did you mean by <span className="text-white font-semibold">“{query.trim()}”</span>?
          </p>
          <div className="flex flex-col gap-2">
            {clarification.options.map((option) => (
              <button
                key={option.label}
                onClick={() => chooseClarification(option)}
                disabled={loading}
                className="bg-[#1a1d27] hover:bg-[#2a2d3e] border border-[#2e3142]
                           text-left rounded-xl px-5 py-3 transition
                           hover:border-indigo-500 disabled:opacity-40"
              >
                <div className="text-white font-medium">{option.label}</div>
                {option.hint && (
                  <div className="text-slate-400 text-sm mt-0.5">{option.hint}</div>
                )}
              </button>
            ))}
          </div>
          <button
            onClick={() => { setClarification(null); setLoading(false) }}
            disabled={loading}
            className="mt-4 text-slate-500 hover:text-slate-300 text-sm mx-auto block
                       disabled:opacity-40"
          >
            ← Back
          </button>
          {error && (
            <p className="text-red-400 mt-3 text-sm text-center" role="alert">{error}</p>
          )}
        </div>
      ) : (
        <>
          {/* Search form */}
          <form onSubmit={onSubmit} className="w-full max-w-2xl animate-fade-in">
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. Artificial General Intelligence, Quantum Computing…"
                disabled={loading}
                className="flex-1 bg-[#1a1d27] border border-[#2e3142] rounded-xl px-5 py-4
                           text-white placeholder-slate-500 text-base outline-none
                           focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30
                           disabled:opacity-50 transition"
                aria-label="Topic to explore"
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-500/40
                           text-white font-semibold px-7 py-4 rounded-xl transition
                           disabled:cursor-not-allowed"
              >
                {loading ? 'Starting…' : 'Explore →'}
              </button>
            </div>
            {error && (
              <p className="text-red-400 mt-3 text-sm text-center" role="alert">{error}</p>
            )}
          </form>

          {/* Example chips */}
          <div className="mt-8 flex flex-wrap gap-2 justify-center animate-fade-in"
               aria-label="Example topics">
            {EXAMPLE_TOPICS.map((t) => (
              <button
                key={t}
                onClick={() => { setQuery(t); submit(t) }}
                disabled={loading}
                className="bg-[#1a1d27] hover:bg-[#2a2d3e] border border-[#2e3142]
                           text-slate-300 text-sm px-4 py-2 rounded-full transition
                           hover:border-indigo-500 hover:text-white disabled:opacity-40"
              >
                {t}
              </button>
            ))}
          </div>
        </>
      )}

      <p className="mt-16 text-slate-600 text-xs text-center">
        Powered by NLP · Knowledge Graphs · Claude AI
      </p>
    </div>
  )
}
