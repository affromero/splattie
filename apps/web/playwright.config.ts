import { defineConfig, devices } from '@playwright/test';

const port = process.env.SPLATTIE_WEB_E2E_PORT ?? '4001';
const baseURL = `http://localhost:${port}`;

/**
 * E2E config for the localhost body/head widget tests. Runs the real Next dev
 * server and drives chromium with software WebGL (swiftshader) so the Spark
 * gaussian-splat renderer works headless in CI.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL,
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
    command: `npx next dev -p ${port}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
