import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Listen on all network interfaces  
    port: 5173,
    watch: {
      usePolling: true, // Needed for Docker
    },
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://backend:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path, // Don't rewrite the path
      }
    }
  }
})
