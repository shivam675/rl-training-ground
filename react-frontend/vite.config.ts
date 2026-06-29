import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// ponytail: no proxy — backend CORS is wide open, so the app talks to
// VITE_API_URL directly. Set it in .env to point elsewhere.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: false },
})
