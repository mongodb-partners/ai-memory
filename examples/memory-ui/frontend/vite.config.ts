import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/',
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // The frontend always calls `/api/...`, so one code path serves dev and
      // prod. In dev this proxy strips the prefix; in prod a reverse proxy does.
      '/api': {
        target: 'http://localhost:8100',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // SSE dies without this. Vite's default proxy buffers responses, so
        // tokens arrive in one burst at the end of the turn — the stream still
        // "works" but the demo's whole point (watching it think) is gone.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache, no-transform';
          });
        },
      },
    },
  },
});
