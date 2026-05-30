import { defineConfig, devices } from '@playwright/test';

/**
 * E2E config for the localhost body/head widget tests. Runs the real Next dev
 * server on :4001 and drives chromium with software WebGL (swiftshader) so the
 * Spark gaussian-splat renderer works headless in CI.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4001',
    ...devices['Desktop Chrome'],
    launchOptions: {
      args: [
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--enable-unsafe-swiftshader',
        '--ignore-gpu-blocklist',
      ],
    },
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:4001',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
