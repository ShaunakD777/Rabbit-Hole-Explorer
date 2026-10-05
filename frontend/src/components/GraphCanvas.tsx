import { useEffect, useRef, useState } from 'react'
import cytoscape from 'cytoscape'
// @ts-ignore – fcose has no bundled types
import fcose from 'cytoscape-fcose'
import { useAppStore } from '../store/appStore'
import type { EdgeData, NodeData } from '../types'

cytoscape.use(fcose)

const RELATION_COLORS: Record<string, string> = {
  prerequisite_of: '#6366f1',
  enables:         '#22c55e',
  subtopic_of:     '#94a3b8',
  related_to:      '#475569',
  breaks:          '#ef4444',
}

function nodeColor(node: NodeData, status: string, isSelected: boolean): string {
  if (isSelected)     return '#6366f1'
  if (status === 'learned')     return '#22c55e'
  if (status === 'in_progress') return '#f59e0b'
  if (node.is_trending)         return '#f97316'
  return '#334155'
}

// The learning path's next step: the first node on it not yet marked learned.
function nextStepId(learningPath: string[], progress: Record<string, string>): string | null {
  return learningPath.find((id) => progress[id] !== 'learned') ?? null
}

// By default only the edges that tell a learner what to study first are drawn:
// prerequisite_of, plus whatever attaches an expanded (depth >= 1) node to the graph
// so it doesn't float. Every typed relation at once (labels and all) read as an
// unlabelled tangle. If a graph has no prerequisite edges at all, everything is shown.
function isEdgeShownByDefault(edge: EdgeData, depthById: Map<string, number>, hasPrereqs: boolean) {
  if (!hasPrereqs || edge.relation_type === 'prerequisite_of') return true
  return (depthById.get(edge.source_node_id) ?? 0) > 0 || (depthById.get(edge.target_node_id) ?? 0) > 0
}

export default function GraphCanvas() {
  const { graphData, selectedNodeId, progress, learningPath } = useAppStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [showAllRelations, setShowAllRelations] = useState(false)

  const hasPrereqs = !!graphData?.edges.some((e) => e.relation_type === 'prerequisite_of')
  const depthById = new Map((graphData?.nodes ?? []).map((n) => [n.id, n.depth_level]))
  const hasHiddenRelations = !!graphData && hasPrereqs &&
    graphData.edges.some((e) => !isEdgeShownByDefault(e, depthById, hasPrereqs))

  // Initialise Cytoscape instance ONCE
  useEffect(() => {
    if (!containerRef.current) return

    const cy = cytoscape({
      container: containerRef.current,
      wheelSensitivity: 0.3,
    })

    cyRef.current = cy

    // Click on node → select
    cy.on('tap', 'node', (evt) => {
      const id = evt.target.id()
      const { selectedNodeId: currentSelected, selectNode: doSelect } = useAppStore.getState()
      doSelect(id === currentSelected ? null : id)
    })

    // Click on background → deselect
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        useAppStore.getState().selectNode(null)
      }
    })

    // Edge relation labels on hover only. Cytoscape has no :hover selector state --
    // the old 'edge:hover' style rule wasn't a hover rule at all, and every edge
    // label rendered permanently -- so hover is tracked with a class instead.
    cy.on('mouseover', 'edge', (evt) => { evt.target.addClass('hovered') })
    cy.on('mouseout', 'edge', (evt) => { evt.target.removeClass('hovered') })

    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [])

  // Sync graphData to Cytoscape instance incrementally
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !graphData) return

    const newGraphNodeIds = new Set(graphData.nodes.map(n => n.id))
    const nodesToRemove = cy.nodes().filter(n => !newGraphNodeIds.has(n.id()))
    if (nodesToRemove.length > 0) cy.remove(nodesToRemove)

    const newGraphEdgeIds = new Set(graphData.edges.map(e => e.id))
    const edgesToRemove = cy.edges().filter(e => !newGraphEdgeIds.has(e.id()))
    if (edgesToRemove.length > 0) cy.remove(edgesToRemove)

    const existingNodeIds = new Set(cy.nodes().map(n => n.id()))
    const existingEdgeIds = new Set(cy.edges().map(e => e.id()))

    const newNodes = graphData.nodes.filter(n => !existingNodeIds.has(n.id)).map((n) => ({
      group: 'nodes' as const,
      data: {
        importance: n.importance_score,
        ...n,
      },
    }))

    const newEdges = graphData.edges.filter(e => !existingEdgeIds.has(e.id)).map((e) => ({
      group: 'edges' as const,
      data: {
        id: e.id,
        source: e.source_node_id,
        target: e.target_node_id,
        relation_type: e.relation_type,
        confidence: e.confidence,
      },
    }))

    if (newNodes.length > 0 || newEdges.length > 0) {
      cy.add([...newNodes, ...newEdges])

      cy.layout({
        name: 'fcose',
        quality: 'proof',
        animate: true,
        animationDuration: 600,
        randomize: !(existingNodeIds.size > 0),
        nodeSeparation: 120,
        idealEdgeLength: 180,
      } as never).run()
    }
  }, [graphData])

  // Re-style when selection / progress changes
  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !graphData) return
    // Cytoscape's Core has no setStyle() method -- the real API is cy.style(sheet),
    // which is what would actually apply the re-style below. Without this fix,
    // clicking any node would throw "cy.setStyle is not a function" and abort the
    // whole effect, so selection/progress highlighting never rendered at all.
    cy.style(buildStyle(graphData, progress, selectedNodeId, learningPath, showAllRelations))

    // Pan to selected node
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId)
      if (node.length) {
        cy.animate({ center: { eles: node }, zoom: cy.zoom() < 1 ? 1 : cy.zoom() },
                   { duration: 300 })
      }
    }
  }, [selectedNodeId, progress, graphData, learningPath, showAllRelations])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} id="cy" aria-label="Knowledge graph canvas"
           role="img" className="w-full h-full" />

      {/* Legend: how to read the graph as a learning route */}
      <div className="absolute top-3 left-3 bg-[#1a1d27]/90 border border-[#2e3142] rounded-lg
                      px-3 py-2 text-xs text-slate-400 space-y-1 pointer-events-auto">
        <p><span className="text-slate-200 font-medium">1, 2, 3…</span> suggested learning order</p>
        <p><span className="text-amber-400 font-medium">◎ ring</span> your next step</p>
        {hasPrereqs && (
          <p><span style={{ color: RELATION_COLORS.prerequisite_of }} className="font-medium">A → B</span>
             {' '}learn A before B</p>
        )}
        {hasHiddenRelations && (
          <label className="flex items-center gap-1.5 pt-1 cursor-pointer select-none">
            <input type="checkbox" checked={showAllRelations}
                   onChange={(e) => setShowAllRelations(e.target.checked)} />
            Show all relation types
          </label>
        )}
      </div>
    </div>
  )
}

function buildStyle(
  graphData: { nodes: NodeData[]; edges: EdgeData[] },
  progress: Record<string, string>,
  selectedNodeId: string | null,
  learningPath: string[],
  showAllRelations: boolean,
): cytoscape.StylesheetStyle[] {
  const nodes = graphData.nodes
  const stepById = new Map(learningPath.map((id, i) => [id, i + 1]))
  const nextId = nextStepId(learningPath, progress)
  const depthById = new Map(nodes.map((n) => [n.id, n.depth_level]))
  const hasPrereqs = graphData.edges.some((e) => e.relation_type === 'prerequisite_of')
  const edgeById = new Map(graphData.edges.map((e) => [e.id, e]))

  return [
    {
      selector: 'node',
      style: {
        'background-color': (ele: cytoscape.NodeSingular) => {
          const n = nodes.find((x) => x.id === ele.id())!
          return n ? nodeColor(n, progress[n.id] ?? 'not_started', ele.id() === selectedNodeId) : '#334155'
        },
        // Step number on the node itself, matching the "Start Here" panel.
        'label': (ele: cytoscape.NodeSingular) => {
          const step = stepById.get(ele.id())
          const label = ele.data('label') as string
          return step ? `${step}. ${label}` : label
        },
        'color': '#e2e8f0',
        'font-size': '11px',
        'font-family': 'Inter, system-ui, sans-serif',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 6,
        'text-max-width': '120px',
        'text-wrap': 'ellipsis',
        'width': (ele: cytoscape.NodeSingular) => {
          const imp = (ele.data('importance_score') as number) ?? 1
          return Math.max(30, Math.min(60, 30 + imp * 10)) + 'px'
        },
        'height': (ele: cytoscape.NodeSingular) => {
          const imp = (ele.data('importance_score') as number) ?? 1
          return Math.max(30, Math.min(60, 30 + imp * 10)) + 'px'
        },
        // Selection wins; otherwise the next step on the path gets an amber ring so
        // the canvas itself answers "where do I start / what's next?".
        'border-width': (ele: cytoscape.NodeSingular) =>
          ele.id() === selectedNodeId ? 3 : ele.id() === nextId ? 4 : 1.5,
        'border-color': (ele: cytoscape.NodeSingular) =>
          ele.id() === selectedNodeId ? '#818cf8' : ele.id() === nextId ? '#fbbf24' : '#475569',
        'transition-property': 'background-color, border-color, width, height',
        'transition-duration': '200ms',
      } as never,
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#818cf8',
      } as never,
    },
    {
      selector: 'edge',
      style: {
        'width': (ele: cytoscape.EdgeSingular) =>
          Math.max(1, ((ele.data('confidence') as number) ?? 0.5) * 3),
        'line-color': (ele: cytoscape.EdgeSingular) =>
          RELATION_COLORS[ele.data('relation_type') as string] ?? '#475569',
        'target-arrow-color': (ele: cytoscape.EdgeSingular) =>
          RELATION_COLORS[ele.data('relation_type') as string] ?? '#475569',
        'target-arrow-shape': (ele: cytoscape.EdgeSingular) =>
          ['prerequisite_of', 'enables', 'subtopic_of'].includes(ele.data('relation_type'))
            ? 'triangle' : 'none',
        'curve-style': 'bezier',
        'line-style': (ele: cytoscape.EdgeSingular) =>
          ele.data('relation_type') === 'related_to' ? 'dashed' : 'solid',
        'opacity': 0.7,
        'display': (ele: cytoscape.EdgeSingular) => {
          const edge = edgeById.get(ele.id())
          if (showAllRelations || !edge) return 'element'
          return isEdgeShownByDefault(edge, depthById, hasPrereqs) ? 'element' : 'none'
        },
      } as never,
    },
    {
      selector: 'edge.hovered',
      style: {
        opacity: 1,
        'label': (ele: cytoscape.EdgeSingular) =>
          (ele.data('relation_type') as string).replace(/_/g, ' '),
        'font-size': '9px',
        'color': '#94a3b8',
        'text-background-color': '#0f1117',
        'text-background-opacity': 0.8,
        'text-background-padding': '2px',
      } as never,
    },
  ]
}
