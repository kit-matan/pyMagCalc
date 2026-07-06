import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import os from 'node:os'

// https://vite.dev/config/
export default defineConfig({
  // Keep Vite's dependency-optimization cache on the local disk. This repo lives
  // on a cloud-synced (~/Library/CloudStorage) filesystem whose per-file access
  // is slow, which otherwise makes the dev-server cold start take ~20 s and the
  // browser open to a blank page. See examples/CCSF/README or the project docs.
  cacheDir: `${os.homedir()}/.magcalc-local/vite-cache`,
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/files': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        changeOrigin: true,
        ws: true
      }
    }
  }
})
