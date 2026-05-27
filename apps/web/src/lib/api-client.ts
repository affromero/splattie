import type { GenerationProgress, GenerationResult, HealthResponse, SegmentResponse } from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export { API_URL };

export async function segmentImage(image: File): Promise<SegmentResponse> {
  const formData = new FormData();
  formData.append('image', image);

  const res = await fetch(`${API_URL}/segment`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Segmentation failed: ${res.statusText}`);
  }

  return res.json();
}

export async function generateFromUpload(image: File): Promise<{
  modelId: string;
  zipUrl: string;
  inferenceSeconds: number;
}> {
  const formData = new FormData();
  formData.append('image', image);

  const res = await fetch(`${API_URL}/generate-from-upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }

  return res.json();
}

export async function generateHead(
  imageUrl: string,
  maskUrl: string,
  onProgress: (progress: GenerationProgress) => void
): Promise<GenerationResult> {
  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_url: imageUrl, mask_url: maskUrl }),
  });

  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let result: GenerationResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    for (const line of text.split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const data = JSON.parse(line.slice(6));

      if ('pct' in data) {
        onProgress(data as GenerationProgress);
      } else if ('modelId' in data) {
        result = data as GenerationResult;
      }
    }
  }

  if (!result) throw new Error('No generation result received');
  return result;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}
