import { expect, test } from '@playwright/test';

/**
 * End-to-end: a body `.splattie` loads in the widget on localhost, renders via
 * Spark SMPL-X skinning, and reacts to the cursor (head/torso look-at). This is
 * the localhost body proof — it exercises the real CDN widget bundle + the body
 * bundle the LHM pipeline produced.
 */
const BODY_SRC = '/demos/bodies/6668809.splattie';

test('body .splattie renders and follows the cursor (look-at)', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));

  await page.goto(`/editor.html?src=${BODY_SRC}`);

  // Widget sets _stateMachine after the bundle's splatload — wait for it.
  await page.waitForFunction(
    () => {
      const w = document.querySelector('splattie-widget') as unknown as { _stateMachine?: unknown };
      return Boolean(w && w._stateMachine);
    },
    { timeout: 60_000 },
  );

  const canvas = page.locator('splattie-widget canvas');
  await expect(canvas).toBeVisible();

  // 1.D implemented the body skinning — the old stub threw this.
  expect(pageErrors.join('\n')).not.toContain('body skinning not implemented');

  const box = (await canvas.boundingBox())!;
  expect(box.width).toBeGreaterThan(0);

  // Wait until the body actually renders (Spark needs a few frames). A blank
  // canvas screenshots tiny (~3.5KB); a rendered 40k-gaussian body is much larger.
  await expect
    .poll(async () => (await canvas.screenshot()).length, { timeout: 20_000, intervals: [300, 300, 500] })
    .toBeGreaterThan(40_000);

  // Look-at: the rendered figure must change as the cursor moves across it.
  await page.mouse.move(box.x + box.width * 0.18, box.y + box.height * 0.4);
  await page.waitForTimeout(700);
  const lookLeft = await canvas.screenshot({ path: '/tmp/body-look-left.png' });

  await page.mouse.move(box.x + box.width * 0.82, box.y + box.height * 0.4);
  await page.waitForTimeout(700);
  const lookRight = await canvas.screenshot({ path: '/tmp/body-look-right.png' });

  // Both gaze directions render a body, and the figure turns toward the cursor
  // (a blank or un-animated body would be tiny or byte-identical).
  expect(lookLeft.length).toBeGreaterThan(40_000);
  expect(lookRight.length).toBeGreaterThan(40_000);
  expect(Buffer.compare(lookLeft, lookRight)).not.toBe(0);

  // Inline editor panel is body-aware (1.E): head/torso tracking controls, no FLAME
  // blendshapes, and no head-config crash.
  await expect(page.locator('#panel')).toContainText('torso');
  expect(pageErrors.join('\n')).not.toContain('Cannot convert undefined');
});

test('body editor poses a limb via IK drag handles', async ({ page }) => {
  await page.goto(`/editor.html?src=${BODY_SRC}`);
  await page.waitForFunction(
    () => {
      const w = document.querySelector('splattie-widget') as unknown as { _stateMachine?: unknown };
      return Boolean(w && w._stateMachine);
    },
    { timeout: 60_000 },
  );
  const canvas = page.locator('splattie-widget canvas');
  await expect
    .poll(async () => (await canvas.screenshot()).length, { timeout: 20_000, intervals: [300, 300, 500] })
    .toBeGreaterThan(40_000);

  // Enter pose mode → a draggable handle appears at each limb end-effector.
  await page.locator('.pose-btn').first().click();
  await expect(page.locator('.ik-handle')).toHaveCount(4);

  // Dragging a handle solves two-bone IK and writes the limb's joint rotations
  // into the edited state's pose.
  const box = (await page.locator('.ik-handle').first().boundingBox())!;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 30, cy - 200, { steps: 16 });
  await page.mouse.up();
  await page.waitForTimeout(400);

  const posedJoints = await page.evaluate(() => {
    const w = document.querySelector('splattie-widget') as unknown as {
      _stateMachine?: { currentFrame?: { pose?: Record<string, unknown> } };
    };
    return Object.keys(w._stateMachine?.currentFrame?.pose ?? {});
  });
  expect(posedJoints).toContain('L_Shoulder');
  expect(posedJoints).toContain('L_Elbow');
  // The body still renders after posing (no crash, no blank canvas).
  await expect.poll(async () => (await canvas.screenshot()).length).toBeGreaterThan(40_000);
});
