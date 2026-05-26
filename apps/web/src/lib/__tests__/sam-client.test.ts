import { beforeAll, describe, expect, it } from 'vitest';
import { createMaskOverlay, segmentWithSAM } from '../sam-client';

class ImageDataPolyfill {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, height?: number) {
    if (typeof dataOrWidth === 'number') {
      this.width = dataOrWidth;
      this.height = widthOrHeight;
      this.data = new Uint8ClampedArray(this.width * this.height * 4);
    } else {
      this.data = dataOrWidth;
      this.width = widthOrHeight;
      this.height = height ?? dataOrWidth.length / (4 * widthOrHeight);
    }
  }
}

beforeAll(() => {
  if (typeof globalThis.ImageData === 'undefined') {
    // @ts-expect-error polyfill for jsdom
    globalThis.ImageData = ImageDataPolyfill;
  }
});

function makeImageData(w: number, h: number, fill: number = 128): ImageData {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = fill;
    data[i + 1] = fill;
    data[i + 2] = fill;
    data[i + 3] = 255;
  }
  return new ImageData(data, w, h);
}

describe('segmentWithSAM', () => {
  it('returns an empty mask with no points', async () => {
    const image = makeImageData(100, 100);
    const result = await segmentWithSAM(image, []);
    expect(result.mask.width).toBe(100);
    expect(result.mask.height).toBe(100);
  });

  it('creates a mask centered on the click point', async () => {
    const image = makeImageData(200, 200);
    const result = await segmentWithSAM(image, [{ x: 100, y: 100, label: 1 }]);

    const centerIdx = (100 * 200 + 100) * 4;
    expect(result.mask.data[centerIdx + 3]).toBe(255);

    expect(result.bbox[0]).toBeGreaterThanOrEqual(0);
    expect(result.bbox[1]).toBeGreaterThanOrEqual(0);
    expect(result.bbox[2]).toBeGreaterThan(0);
    expect(result.bbox[3]).toBeGreaterThan(0);
  });

  it('mask is not all white', async () => {
    const image = makeImageData(200, 200);
    const result = await segmentWithSAM(image, [{ x: 100, y: 100, label: 1 }]);

    let transparent = 0;
    for (let i = 0; i < result.mask.data.length; i += 4) {
      if (result.mask.data[i + 3] === 0) transparent++;
    }
    expect(transparent).toBeGreaterThan(0);
  });
});

describe('createMaskOverlay', () => {
  it('dims pixels outside the mask', () => {
    const original = makeImageData(10, 10, 200);
    const mask = makeImageData(10, 10, 0);
    for (let i = 0; i < mask.data.length; i += 4) {
      mask.data[i + 3] = 0;
    }

    const overlay = createMaskOverlay(original, mask, 0.5);

    expect(overlay.data[0]).toBe(100);
    expect(overlay.data[1]).toBe(100);
    expect(overlay.data[2]).toBe(100);
  });

  it('preserves pixels inside the mask', () => {
    const original = makeImageData(10, 10, 200);
    const mask = makeImageData(10, 10, 255);

    const overlay = createMaskOverlay(original, mask, 0.5);

    expect(overlay.data[0]).toBe(200);
    expect(overlay.data[1]).toBe(200);
  });
});
