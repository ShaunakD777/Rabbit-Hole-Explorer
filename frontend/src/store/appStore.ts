import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { GraphData, NodeData, ProgressStatus } from '../types'

interface AppState {
  // Topic / graph
  topicId: string | null
  topicLabel: string
  graphId: string | null
  graphData: GraphData | null
  learningPath: string[]

  // Pipeline status
  pipelineStep: number        // 0=idle 1=searching 2=extracting 3=building 4=ready
  pipelineMessage: string
  isError: boolean

  // UI state
  selectedNodeId: string | null
  expandingNodeId: string | null
  isDetailPanelOpen: boolean
  isTrendsPanelOpen: boolean
  isProgressPanelOpen: boolean

  // Progress tracking (nodeId → status)
  progress: Record<string, ProgressStatus>

  // Actions
  setTopic: (id: string, label: string) => void
  setGraphData: (data: GraphData, path: string[]) => void
  setPipelineStep: (step: number, message: string) => void
  setError: (msg: string) => void
  selectNode: (id: string | null) => void
  setExpandingNode: (id: string | null) => void
  closeDetailPanel: () => void
  toggleTrendsPanel: () => void
  toggleProgressPanel: () => void
  updateProgress: (nodeId: string, status: ProgressStatus) => void
  mergeNewNodes: (parentNodeId: string, newNodes: NodeData[], newEdges?: any[]) => void
  reset: () => void
}

const initialState = {
  topicId: null,
  topicLabel: '',
  graphId: null,
  graphData: null,
  learningPath: [],
  pipelineStep: 0,
  pipelineMessage: '',
  isError: false,
  selectedNodeId: null,
  expandingNodeId: null,
  isDetailPanelOpen: false,
  isTrendsPanelOpen: false,
  isProgressPanelOpen: false,
  progress: {},
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      ...initialState,

      setTopic: (id, label) =>
        set({ topicId: id, topicLabel: label, pipelineStep: 1,
              pipelineMessage: 'Searching sources…', isError: false }),

      setGraphData: (data, path) =>
        set({ graphData: data, learningPath: path, pipelineStep: 4,
              pipelineMessage: 'Ready', graphId: data.graph_id }),

      setPipelineStep: (step, message) =>
        set({ pipelineStep: step, pipelineMessage: message }),

      setError: (msg) =>
        set({ isError: true, pipelineMessage: msg, pipelineStep: 0 }),

      selectNode: (id) =>
        set({ selectedNodeId: id, isDetailPanelOpen: id !== null }),

      setExpandingNode: (id) => set({ expandingNodeId: id }),

      closeDetailPanel: () =>
        set({ isDetailPanelOpen: false, selectedNodeId: null }),

      toggleTrendsPanel: () =>
        set((s) => ({ isTrendsPanelOpen: !s.isTrendsPanelOpen,
                      isProgressPanelOpen: false })),

      toggleProgressPanel: () =>
        set((s) => ({ isProgressPanelOpen: !s.isProgressPanelOpen,
                      isTrendsPanelOpen: false })),

      updateProgress: (nodeId, status) =>
        set((s) => ({ progress: { ...s.progress, [nodeId]: status } })),

      mergeNewNodes: (_parentNodeId, newNodes, newEdges) =>
        set((s) => {
          if (!s.graphData) return {}
          const existingNodes = new Set(s.graphData.nodes.map((n) => n.id))
          const freshNodes = newNodes.filter((n) => !existingNodes.has(n.id))

          const existingEdges = new Set(s.graphData.edges.map((e) => e.id))
          const freshEdges = (newEdges || []).filter((e) => !existingEdges.has(e.id))

          return {
            graphData: {
              ...s.graphData,
              nodes: [...s.graphData.nodes, ...freshNodes],
              edges: [...s.graphData.edges, ...freshEdges],
            },
            expandingNodeId: null,
          }
        }),

      reset: () => set(initialState),
    }),
    {
      // Survive a page refresh: previously all state (including topicId) was
      // in-memory only, so refreshing while viewing a graph threw the user back
      // to the landing page with no way to recover the topic they were on.
      name: 'rabbit-hole-explorer-store',
      partialize: (state) => ({
        topicId: state.topicId,
        topicLabel: state.topicLabel,
        graphId: state.graphId,
        graphData: state.graphData,
        learningPath: state.learningPath,
        pipelineStep: state.pipelineStep,
        pipelineMessage: state.pipelineMessage,
        progress: state.progress,
      }),
    }
  )
)
