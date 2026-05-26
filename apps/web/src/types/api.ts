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
  flameParamsUrl: string;
}

export interface MethodInfo {
  id: string;
  name: string;
  description: string;
  paperUrl: string;
  repoUrl: string;
}

export interface HealthResponse {
  status: string;
  gpu: string;
  methodsLoaded: string[];
}
