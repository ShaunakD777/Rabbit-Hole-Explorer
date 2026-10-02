import { useAppStore } from '../store/appStore'

const STEPS = [
  'Understanding topic',
  'Searching sources',
  'Extracting concepts',
  'Building knowledge graph',
]

export default function PipelineProgress() {
  const { pipelineStep, pipelineMessage, isError } = useAppStore()

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-4">
      <div className="w-full max-w-md">
        <h3 className="text-white text-lg font-semibold mb-6 text-center">
          {isError ? '⚠️ Something went wrong' : '🔍 Exploring the rabbit hole…'}
        </h3>

        <ol className="space-y-3" aria-label="Pipeline steps">
          {STEPS.map((label, i) => {
            const stepNum = i + 1
            const done = pipelineStep > stepNum
            const active = pipelineStep === stepNum
            return (
              <li key={label}
                  className={`flex items-center gap-3 text-sm transition
                    ${done ? 'text-green-400' : active ? 'text-white' : 'text-slate-600'}`}
                  aria-current={active ? 'step' : undefined}>
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs
                  font-bold border-2 shrink-0 transition
                  ${done ? 'bg-green-500 border-green-500 text-white'
                  : active ? 'border-indigo-400 text-indigo-400 animate-pulse-glow'
                  : 'border-slate-700 text-slate-700'}`}>
                  {done ? '✓' : stepNum}
                </span>
                {label}
                {active && <span className="ml-auto text-xs text-indigo-400 animate-pulse">●</span>}
              </li>
            )
          })}
        </ol>

        {isError && (
          <p className="mt-4 text-red-400 text-sm text-center" role="alert">
            {pipelineMessage}
          </p>
        )}

        {!isError && pipelineMessage && (
          <p className="mt-4 text-slate-400 text-xs text-center">{pipelineMessage}</p>
        )}
      </div>
    </div>
  )
}
