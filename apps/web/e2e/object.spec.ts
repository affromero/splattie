import { expect, test } from '@playwright/test';

/**
 * End-to-end object proof: a generated arbitrary-object `.splattie` bundle loads
 * through the real editor page and renders through the object-aware widget.
 */
const OBJECT_SRC = '/demos/objects/o3.splattie';

test('object .splattie renders in the widget on localhost', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));

  await page.goto(`/editor.html?src=${OBJECT_SRC}`);

  await page.waitForFunction(
    () => {
      const w = document.querySelector('splattie-widget') as unknown as {
        _stateMachine?: unknown;
        assetType?: string;
      };
      return Boolean(w && w._stateMachine && w.assetType === 'object');
    },
    { timeout: 60_000 },
  );

  const canvas = page.locator('splattie-widget canvas');
  await expect(canvas).toBeVisible();
  // The compact mannequin demo occupies much less of the frame than the body/head
  // demos, so its PNG screenshot is smaller while still proving the canvas is not
  // the blank ~3.5KB WebGL clear frame.
  await expect
    .poll(async () => (await canvas.screenshot()).length, { timeout: 25_000, intervals: [300, 300, 500] })
    .toBeGreaterThan(8_000);

  const panel = page.locator('#panel');
  await expect(panel).toContainText('camera');
  await expect(panel).toContainText('rotation');
  await expect(panel).toContainText('follow');
  await expect(panel).toContainText('rig');
  await expect(panel).toContainText('root');
  await expect(panel).toContainText('joints');
  await expect(page.locator('.rig-joint').first()).toBeVisible();
  await expect(page.locator('.rig-axis input[type="range"]').first()).toBeVisible();
  await expect(page.locator('.pose-btn').filter({ hasText: 'edit pose' })).toBeVisible();
  await expect(panel).not.toContainText('smile');
  await expect(panel).not.toContainText('jawOpen');
  await expect(panel).not.toContainText('tracking');
  await expect(panel).not.toContainText('torso');

  await page.locator('.pose-btn').filter({ hasText: 'edit pose' }).click();
  await expect(page.locator('.ik-handle').first()).toBeVisible();
  await expect.poll(async () => await page.locator('.rig-lines line').count()).toBeGreaterThan(0);
  const handleBox = (await page.locator('.ik-handle').first().boundingBox())!;
  await page.mouse.move(handleBox.x + handleBox.width / 2, handleBox.y + handleBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(handleBox.x + 32, handleBox.y - 24, { steps: 8 });
  await page.mouse.up();
  await expect
    .poll(async () => {
      return await page.locator('splattie-widget').evaluate((el) => {
        const widget = el as unknown as { _stateMachine?: { currentFrame?: { pose?: Record<string, unknown> } } };
        return Object.keys(widget._stateMachine?.currentFrame?.pose ?? {}).length;
      });
    })
    .toBeGreaterThan(0);

  await page.locator('.rig-axis input[type="range"]').first().evaluate((el) => {
    const input = el as HTMLInputElement;
    input.value = '25';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect
    .poll(async () => {
      return await page.locator('splattie-widget').evaluate((el) => {
        const widget = el as unknown as { _stateMachine?: { currentFrame?: { pose?: Record<string, unknown> } } };
        return Object.keys(widget._stateMachine?.currentFrame?.pose ?? {}).length;
      });
    })
    .toBeGreaterThan(0);

  const errors = pageErrors.join('\n');
  expect(errors).not.toContain('format version mismatch');
  expect(errors).not.toContain('body skinning not implemented');
  expect(errors).not.toContain('Cannot convert');
});
