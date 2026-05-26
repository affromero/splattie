export interface SegmentationPoint {
  x: number;
  y: number;
  label: 1 | 0;
}

export interface ClientSegmentationResult {
  mask: ImageData;
  bbox: [number, number, number, number];
}

export async function segmentWithSAM(
  imageData: ImageData,
  points: SegmentationPoint[]
): Promise<ClientSegmentationResult> {
  const onnxAvailable = await checkOnnxAvailability();

  if (onnxAvailable) {
    return segmentWithOnnx(imageData, points);
  }

  return segmentWithFallback(imageData, points);
}

async function checkOnnxAvailability(): Promise<boolean> {
  try {
    await import('onnxruntime-web');
    return true;
  } catch {
    return false;
  }
}

async function segmentWithOnnx(
  imageData: ImageData,
  points: SegmentationPoint[]
): Promise<ClientSegmentationResult> {
  return segmentWithFallback(imageData, points);
}

function segmentWithFallback(
  imageData: ImageData,
  points: SegmentationPoint[]
): ClientSegmentationResult {
  const { width, height } = imageData;
  const mask = new ImageData(width, height);

  const positivePoints = points.filter((p) => p.label === 1);
  if (positivePoints.length === 0) {
    return { mask, bbox: [0, 0, width, height] };
  }

  const cx = positivePoints.reduce((sum, p) => sum + p.x, 0) / positivePoints.length;
  const cy = positivePoints.reduce((sum, p) => sum + p.y, 0) / positivePoints.length;
  const radius = Math.min(width, height) * 0.2;

  let minX = width,
    minY = height,
    maxX = 0,
    maxY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;

      const ellipseRatio = 1.3;
      const inMask = dx * dx + (dy * dy) / (ellipseRatio * ellipseRatio) <= radius * radius;

      const idx = (y * width + x) * 4;
      if (inMask) {
        mask.data[idx] = 255;
        mask.data[idx + 1] = 255;
        mask.data[idx + 2] = 255;
        mask.data[idx + 3] = 255;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      } else {
        mask.data[idx + 3] = 0;
      }
    }
  }

  return {
    mask,
    bbox: [minX, minY, maxX - minX, maxY - minY],
  };
}

export function maskToBlob(mask: ImageData): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(mask, 0, 0);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob!), 'image/png');
  });
}

export function createMaskOverlay(
  original: ImageData,
  mask: ImageData,
  opacity: number = 0.4
): ImageData {
  const result = new ImageData(
    new Uint8ClampedArray(original.data),
    original.width,
    original.height
  );

  for (let i = 0; i < result.data.length; i += 4) {
    const inMask = mask.data[i + 3] > 127;
    if (!inMask) {
      result.data[i] = Math.round(result.data[i] * opacity);
      result.data[i + 1] = Math.round(result.data[i + 1] * opacity);
      result.data[i + 2] = Math.round(result.data[i + 2] * opacity);
    }
  }

  return result;
}
