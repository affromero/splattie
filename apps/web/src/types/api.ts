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
  spzUrl: string;
  spzSizeBytes: number;
  numGaussians: number;
  methodId: string;
  rigParamsUrl: string;
}

export type AssetType = 'head' | 'body' | 'object';

export interface MethodInfo {
  id: string;
  name: string;
  description: string;
  paperUrl: string;
  repoUrl: string;
  assetType: AssetType;
}

export interface HealthResponse {
  status: string;
  gpu: string;
  methodsLoaded: string[];
}
