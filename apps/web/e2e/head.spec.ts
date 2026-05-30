import { expect, test } from '@playwright/test';

/**
 * Regression: a head `.splattie` still renders in the widget and the inline editor
 * shows FLAME blendshape + eye-tracking controls — the body work (assetType branch,
 * optional tracking.eyes, lerpExpression guard) must not break heads.
 */
const HEAD_SRC = '/demos/heads/h1.splattie';

test('head .splattie renders and the editor shows blendshape controls', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));

  await page.goto(`/editor.html?src=${HEAD_SRC}`);
  await page.waitForFunction(
    () => {
      const w = document.querySelector('splattie-widget') as unknown as { _stateMachine?: unknown };
      return Boolean(w && w._stateMachine);
    },
    { timeout: 60_000 },
  );

  const canvas = page.locator('splattie-widget canvas');
  await expect(canvas).toBeVisible();
  await expect
    .poll(async () => (await canvas.screenshot()).length, { timeout: 20_000, intervals: [300, 300, 500] })
    .toBeGreaterThan(40_000);

  // Head editor: FLAME blendshapes + eye tracking (not body controls).
  await expect(page.locator('#panel')).toContainText('jawOpen');
  await expect(page.locator('#panel')).toContainText('eyes');
  expect(pageErrors.join('\n')).not.toContain('Cannot convert');
  expect(pageErrors.join('\n')).not.toContain('body skinning');
});
