import { useEffect, useRef } from 'react'
import cytoscape from 'cytoscape'
// @ts-ignore – fcose has no bundled types
import fcose from 'cytoscape-fcose'
import { useAppStore } from '../store/appStore'
import type { NodeData } from '../types'

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

export default function GraphCanvas() {
  const { graphData, selectedNodeId, progress, learningPath } = useAppStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)

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
    cy.style(buildStyle(graphData.nodes, progress, selectedNodeId))

    // Pan to selected node
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId)
      if (node.length) {
        cy.animate({ center: { eles: node }, zoom: cy.zoom() < 1 ? 1 : cy.zoom() },
                   { duration: 300 })
      }
    }
  }, [selectedNodeId, progress, graphData])

  // Highlight learning path edges
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    learningPath.forEach((nodeId, i) => {
      const node = cy.getElementById(nodeId)
      if (node.length) {
        node.data('pathIndex', i + 1)
      }
    })
  }, [learningPath])

  return (
    <div ref={containerRef} id="cy" aria-label="Knowledge graph canvas"
         role="img" className="w-full h-full" />
  )
}

function buildStyle(
  nodes: NodeData[],
  progress: Record<string, string>,
  selectedNodeId: string | null,
): cytoscape.StylesheetStyle[] {
  return [
    {
      selector: 'node',
      style: {
        'background-color': (ele: cytoscape.NodeSingular) => {
          const n = nodes.find((x) => x.id === ele.id())!
          return n ? nodeColor(n, progress[n.id] ?? 'not_started', ele.id() === selectedNodeId) : '#334155'
        },
        'label': 'data(label)',
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
        'border-width': (ele: cytoscape.NodeSingular) =>
          ele.id() === selectedNodeId ? 3 : 1.5,
        'border-color': (ele: cytoscape.NodeSingular) =>
          ele.id() === selectedNodeId ? '#818cf8' : '#475569',
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
      } as never,
    },
    {
      selector: 'edge:hover',
      style: { opacity: 1, 'label': 'data(relation_type)', 'font-size': '9px',
               'color': '#94a3b8' } as never,
    },
  ]
}
