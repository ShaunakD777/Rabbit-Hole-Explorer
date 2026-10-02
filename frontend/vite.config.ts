import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Not currently exercised: client.ts and useWebSocket.ts both build absolute
    // URLs from VITE_API_URL/VITE_WS_URL, so the browser talks to the `backend`
    // container's published port directly and never sends a relative /api or /ws
    // request through this dev server. Kept correct (pointed at the `backend`
    // service, not `localhost`, which would be wrong from inside this container)
    // in case relative requests are ever introduced.
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://backend:8000',
        ws: true,
      },
    },
  },
})
