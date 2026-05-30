import react from '@vitejs/plugin-react';
import path from 'path';
import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    // Playwright specs in e2e/ run via `test:e2e` (real browser), not vitest.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
});
