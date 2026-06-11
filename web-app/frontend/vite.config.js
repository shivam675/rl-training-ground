import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /ws and /assets to the FastAPI backend so the app is same-origin in dev
// (no CORS, and the WebSocket / URDF fetches "just work" with relative URLs).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': { target: 'ws://localhost:8000', ws: true },
      '/assets': { target: 'http://localhost:8000', changeOrigin: true },
      '/loaded': { target: 'http://localhost:8000', changeOrigin: true },
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
