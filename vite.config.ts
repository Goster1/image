import { defineConfig, type Plugin } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { resolve } from 'node:path';
import { renameSync, rmSync } from 'node:fs';

// Vite emits HTML at its source-relative path; the manifest and service
// worker expect monitor.html at the dist root.
function flattenMonitorHtml(): Plugin {
  return {
    name: 'flatten-monitor-html',
    closeBundle() {
      const dist = resolve(__dirname, 'dist');
      renameSync(
        resolve(dist, 'extension/src/monitor/monitor.html'),
        resolve(dist, 'monitor.html'),
      );
      rmSync(resolve(dist, 'extension'), { recursive: true, force: true });
    },
  };
}

// Builds the extension pages (ES modules): monitor page + MV3 module service worker.
// The content script needs IIFE format and is built by vite.content.config.ts.
export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    rollupOptions: {
      input: {
        monitor: resolve(__dirname, 'extension/src/monitor/monitor.html'),
        'service-worker': resolve(__dirname, 'extension/src/background/service-worker.ts'),
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  plugins: [
    flattenMonitorHtml(),
    viteStaticCopy({
      targets: [
        { src: 'extension/manifest.json', dest: '.' },
        { src: 'extension/public/*', dest: '.' },
        {
          src: 'node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.js',
          dest: 'wasm',
        },
        {
          src: 'node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm',
          dest: 'wasm',
        },
      ],
    }),
  ],
});
