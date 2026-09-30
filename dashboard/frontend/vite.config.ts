import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Network-accessible by design (see ../../.claude/plans — dashboard plan):
// host 0.0.0.0 so this is reachable off the lab PC, not just localhost.
// Dev-time proxy to the FastAPI backend keeps the frontend's fetch/WS calls
// same-origin relative (/api/..., /ws/...) instead of hardcoding a backend
// host that would break the moment this is opened from a different machine.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      // covers both REST (/api/...) and the status websocket (/api/ws/status)
      '/api': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
    },
  },
})
