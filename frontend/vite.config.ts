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
        target: 'http://172.18.0.3:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, _req, _res) => {
            console.log('Proxying request to:', proxyReq.path);
          });
        }
      }
    }
  }
})
