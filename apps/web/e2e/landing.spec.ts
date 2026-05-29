import { expect, test } from '@playwright/test';

/**
 * End-to-end on the real landing page (localhost): both per-category carousels
 * render, and clicking a body avatar opens the inline editor that renders the body
 * via the Spark WebGL widget inside the iframe.
 */
test('landing carousels + inline body editor render on localhost', async ({ page }) => {
  // Stop the carousel marquee (CSS honours prefers-reduced-motion) so cards are
  // stable to click.
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');

  // Both per-category carousels are populated (head + body avatar cards).
  await expect(page.locator('button[aria-label*="head" i]').first()).toBeVisible();
  const bodyCard = page.locator('button[aria-label*="body" i]').first();
  await expect(bodyCard).toBeVisible();

  // Clicking it opens the inline editor pointing at a body bundle.
  await bodyCard.click();
  const iframe = page.locator('iframe[title*="body" i]');
  await expect(iframe).toBeVisible();
  // src is URL-encoded (the .splattie path carries a ?v= cache-bust query).
  await expect(iframe).toHaveAttribute('src', /bodies.*\.splattie/);

  // The body renders inside the inline editor (Spark WebGL within the iframe).
  const canvas = page.frameLocator('iframe[title*="body" i]').locator('splattie-widget canvas');
  await expect(canvas).toBeVisible({ timeout: 60_000 });
  await expect
    .poll(async () => (await canvas.screenshot()).length, { timeout: 25_000, intervals: [500, 500, 1000] })
    .toBeGreaterThan(40_000);

  // With an avatar selected (auto-scroll paused), the carousel stays trackpad-
  // scrollable so you can browse and pick another.
  const carousel = page.locator('[data-category="body"]');
  expect(await carousel.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
  await carousel.evaluate((el) => {
    el.scrollLeft = 250;
  });
  expect(await carousel.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
});
