export interface SegmentResponse {
  maskUrl: string;
  previewUrl: string;
  bbox: [number, number, number, number];
}

export interface GenerationProgress {
  stage: string;
  pct: number;
}

export interface GenerationResult {
  modelId: string;
  splattieUrl: string;
  splattieSizeBytes: number;
  numGaussians: number;
  methodId: string;
}

export type AssetType = 'head' | 'body' | 'object' | 'quadruped_mammal';

// Image->3D-gaussian reconstruction backend (quadruped only; other categories ignore it).
export type ReconstructBackend = 'triposplat' | 'trellis';

export interface MethodInfo {
  id: string;
  name: string;
  description: string;
  paperUrl: string;
  repoUrl: string;
  assetType: AssetType;
}

export interface GpuStatus {
  available: boolean;
  device: string | null;
  vramTotalMb?: number;
  vramUsedMb?: number;
  modelLoaded: boolean;
}

export interface HealthResponse {
  status: string;
  gpu: GpuStatus;
  methodsLoaded: string[];
}
