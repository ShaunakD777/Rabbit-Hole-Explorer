import { useEffect, useRef, useCallback } from 'react'

const WS_BASE = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000'

// Exponential backoff for reconnection: 1s, 2s, 4s, ... capped at 15s. Previously
// there was no reconnection at all -- one dropped connection meant the user missed
// every subsequent pipeline/expansion event until a manual page reload.
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 15000

type MessageHandler = (data: Record<string, unknown>) => void

export function useWebSocket(channelId: string | null, onMessage: MessageHandler) {
  const wsRef = useRef<WebSocket | null>(null)
  const handlerRef = useRef(onMessage)
  handlerRef.current = onMessage

  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const closedByUsRef = useRef(false)

  const connect = useCallback((channel: string) => {
    const url = `${WS_BASE}/ws/channel/${channel}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      console.debug('[WS] Connected to channel', channel)
      reconnectAttemptRef.current = 0
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        handlerRef.current(data)
      } catch {
        // ignore non-JSON pongs
      }
    }

    ws.onerror = (err) => {
      console.warn('[WS] Error on channel', channel, err)
    }

    ws.onclose = () => {
      console.debug('[WS] Disconnected from channel', channel)
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current)
      if (closedByUsRef.current) return

      const attempt = reconnectAttemptRef.current
      const delay = Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS)
      reconnectAttemptRef.current = attempt + 1
      reconnectTimerRef.current = setTimeout(() => connect(channel), delay)
    }

    // Heartbeat ping every 25 s
    pingIntervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping')
    }, 25_000)
  }, [])

  useEffect(() => {
    if (!channelId) return
    closedByUsRef.current = false
    reconnectAttemptRef.current = 0
    connect(channelId)

    return () => {
      closedByUsRef.current = true
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current)
      wsRef.current?.close()
    }
  }, [channelId, connect])
}
