export type RelationType =
  | 'prerequisite_of'
  | 'related_to'
  | 'subtopic_of'
  | 'enables'
  | 'breaks'

export type ProgressStatus = 'not_started' | 'in_progress' | 'learned'

export type SummaryDepth = '2min' | '10min' | 'deepdive'

export type TopicStatus = 'processing' | 'ready' | 'failed' | 'needs_clarification'

export interface ClarificationOption {
  label: string
  clarifying_query: string
  hint: string | null
}

export interface TopicData {
  id: string
  raw_query: string
  normalized_label: string
  status: TopicStatus
  clarification_options: ClarificationOption[] | null
  created_at: string
}

export interface NodeData {
  id: string
  label: string
  aliases: string[]
  description_short: string | null
  category: string | null
  importance_score: number
  is_trending: boolean
  depth_level: number
}

export interface EdgeData {
  id: string
  source_node_id: string
  target_node_id: string
  relation_type: RelationType
  confidence: number
}

export interface GraphData {
  graph_id: string
  topic_id: string
  version: number
  nodes: NodeData[]
  edges: EdgeData[]
  learning_path: string[]
}

export interface SummaryData {
  node_id: string
  depth: SummaryDepth
  content: string
  citations: Array<{ index: number; source_document_id: string }>
  generated_at: string
}

export interface SourceDoc {
  id: string
  source_type: 'wikipedia' | 'youtube' | 'reddit' | 'paper' | 'web'
  url: string
  title: string
  author_or_channel: string | null
  published_at: string | null
  metadata: Record<string, unknown>
}

export interface VideoQuizQuestion {
  question: string
  options: string[]
  correct_index: number
}

export interface VideoExtractionData {
  notes_outline: string
  key_concepts: string[]
  quiz: VideoQuizQuestion[]
}

export interface TrendData {
  id: string
  cluster_label: string
  growth_rate: number | null
  window_start: string | null
  window_end: string | null
  linked_node_id: string | null
}
