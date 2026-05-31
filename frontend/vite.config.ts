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
        // Docker uses 'backend' hostname; local dev uses localhost
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
