import { defineConfig } from 'vite';
import { resolve } from 'node:path';

// Content scripts cannot be ES modules in MV3, so this second pass bundles
// picking-content.ts as a single IIFE into the same dist/ directory.
export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    target: 'es2022',
    rollupOptions: {
      input: resolve(__dirname, 'extension/src/content/picking-content.ts'),
      output: {
        format: 'iife',
        entryFileNames: 'picking-content.js',
      },
    },
  },
});
