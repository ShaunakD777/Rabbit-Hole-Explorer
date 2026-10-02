import { useEffect } from 'react'
import { useAppStore } from '../store/appStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { getTopic, getGraph, getGraphProgress } from '../api/client'
import type { GraphData, NodeData } from '../types'

import TopBar from '../components/TopBar'
import LearningPathPanel from '../components/LearningPathPanel'
import GraphCanvas from '../components/GraphCanvas'
import NodeDetailDrawer from '../components/NodeDetailDrawer'
import TrendsPanel from '../components/TrendsPanel'
import ProgressPanel from '../components/ProgressPanel'
import PipelineProgress from '../components/PipelineProgress'

export default function GraphWorkspace() {
  const {
    topicId, pipelineStep,
    setGraphData, setPipelineStep, setError,
    updateProgress, mergeNewNodes,
  } = useAppStore()

  // Poll for graph once topic transitions to 'ready'
  useEffect(() => {
    if (!topicId || pipelineStep === 4) return
    const poll = async () => {
      try {
        const topicRes = await getTopic(topicId)
        const topic = topicRes.data
        if (topic.status === 'ready') {
          const graphRes = await getGraph(topicId)
          const data: GraphData = graphRes.data
          setGraphData(data, data.learning_path)

          // Load saved progress
          const progRes = await getGraphProgress(data.graph_id)
          progRes.data.forEach((r: { node_id: string; status: string }) => {
            updateProgress(r.node_id, r.status as never)
          })
        } else if (topic.status === 'failed') {
          setError('Pipeline failed. Please try a different topic.')
        }
      } catch {
        // silently retry
      }
    }

    const interval = setInterval(poll, 2500)
    return () => clearInterval(interval)
  }, [topicId, pipelineStep, setGraphData, setError, updateProgress])

  // WebSocket for real-time updates during pipeline
  useWebSocket(topicId, (msg) => {
    const event = msg.event as string
    if (event === 'progress') {
      setPipelineStep(msg.step as number, msg.message as string)
    } else if (event === 'graph_ready') {
      // Graph ready will be caught by the poll above, but we can speed it up:
      getGraph(topicId!).then((res) => {
        setGraphData(res.data, res.data.learning_path)
      }).catch(() => {})
    } else if (event === 'nodes_added') {
      mergeNewNodes(msg.parent_node_id as string, msg.new_nodes as NodeData[], msg.new_edges as any[])
    } else if (event === 'error') {
      setError(msg.message as string)
    }
  })

  return (
    <div className="flex flex-col h-screen overflow-hidden"
         style={{ background: 'var(--bg-canvas)' }}>
      <TopBar />

      <div className="flex flex-1 overflow-hidden">
        {/* Left: Learning Path */}
        <LearningPathPanel />

        {/* Center: Graph canvas */}
        <main className="flex-1 relative overflow-hidden">
          {pipelineStep < 4 ? <PipelineProgress /> : <GraphCanvas />}
        </main>

        {/* Right: Drawer + side panels */}
        <NodeDetailDrawer />
        <TrendsPanel />
        <ProgressPanel />
      </div>
    </div>
  )
}
