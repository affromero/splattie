import type { TrackingConfig } from '../types';

export class CursorTracking {
  getEyeWeights(
    ndcX: number,
    ndcY: number,
    tracking: TrackingConfig,
  ): Record<string, number> {
    const weights: Record<string, number> = {};
    const eyeIntensity = tracking.eyes;
    if (eyeIntensity <= 0) return weights;

    weights.eyeLookInLeft = Math.max(0, -ndcX) * eyeIntensity;
    weights.eyeLookOutLeft = Math.max(0, ndcX) * eyeIntensity;
    weights.eyeLookInRight = Math.max(0, ndcX) * eyeIntensity;
    weights.eyeLookOutRight = Math.max(0, -ndcX) * eyeIntensity;
    weights.eyeLookUpLeft = Math.max(0, ndcY) * 0.8 * eyeIntensity;
    weights.eyeLookUpRight = Math.max(0, ndcY) * 0.8 * eyeIntensity;
    weights.eyeLookDownLeft = Math.max(0, -ndcY) * 0.8 * eyeIntensity;
    weights.eyeLookDownRight = Math.max(0, -ndcY) * 0.8 * eyeIntensity;

    return weights;
  }
}
