import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'SplatWidget',
      formats: ['es', 'umd'],
      fileName: 'splat-widget',
    },
    rollupOptions: {
      external: ['three'],
      output: {
        globals: { three: 'THREE' },
      },
    },
  },
});
