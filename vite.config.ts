import { defineConfig } from 'vite';

// /api/sentence is the large sentence model (server/sentence-api), published on 127.0.0.1:3021
// on the SLI server; dev and preview servers forward to it so the e2e tests use the real API.
const api = { '/api': 'http://127.0.0.1:3021' };

export default defineConfig({
  // Served from the root of a custom domain on GitHub Pages; relative paths also work in subfolders.
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { proxy: api },
  preview: { proxy: api },
  test: { include: process.env.SLI_REPLAY ? ['tests/replay/**/*.test.ts'] : ['tests/unit/**/*.test.ts'], testTimeout: 30000 },
});
