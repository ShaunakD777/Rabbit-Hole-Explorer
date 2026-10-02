import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export const api = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
})

// ── Topics ──────────────────────────────────────────────────────────────────

export const createTopic = (rawQuery: string, priorKnowledge: string[] = []) =>
  api.post('/topics', { raw_query: rawQuery, prior_knowledge: priorKnowledge })

export const clarifyTopic = (topicId: string, chosenQuery: string, priorKnowledge: string[] = []) =>
  api.post(`/topics/${topicId}/clarify`, { chosen_query: chosenQuery, prior_knowledge: priorKnowledge })

export const getTopic = (topicId: string) =>
  api.get(`/topics/${topicId}`)

export const getGraph = (topicId: string) =>
  api.get(`/topics/${topicId}/graph`)

// ── Nodes ────────────────────────────────────────────────────────────────────

export const getNodeSummary = (nodeId: string, depth: string) =>
  api.get(`/nodes/${nodeId}/summary`, { params: { depth } })

export const getNodeSources = (nodeId: string) =>
  api.get(`/nodes/${nodeId}/sources`)

export const expandNode = (nodeId: string) =>
  api.post(`/nodes/${nodeId}/expand`)

// ── Search ───────────────────────────────────────────────────────────────────

export const searchGraph = (graphId: string, q: string) =>
  api.get(`/graph/${graphId}/search`, { params: { q, top_k: 8 } })

// ── Progress ─────────────────────────────────────────────────────────────────

const ANON_USER = '00000000-0000-0000-0000-000000000001'

export const upsertProgress = (nodeId: string, status: string) =>
  api.post(`/progress/${ANON_USER}/${nodeId}`, { status })

export const getGraphProgress = (graphId: string) =>
  api.get(`/progress/${ANON_USER}/graph/${graphId}`)

// ── Trends ───────────────────────────────────────────────────────────────────

export const getTrends = (topicId: string) =>
  api.get(`/trends/${topicId}`)

// ── Videos ───────────────────────────────────────────────────────────────────

export const extractVideo = (sourceDocId: string) =>
  api.post(`/videos/${sourceDocId}/extract`)
