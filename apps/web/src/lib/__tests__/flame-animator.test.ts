import { describe, expect, it } from 'vitest';
import { applyEyePose, generateProceduralHead } from '../flame-animator';
import type { EyePose } from '@/types/gaze';

describe('generateProceduralHead', () => {
  it('generates the requested number of gaussians', () => {
    const { positions, colors, flameParams } = generateProceduralHead(1000);
    expect(positions.length).toBe(3000);
    expect(colors.length).toBe(4000);
    expect(flameParams.numGaussians).toBe(1000);
  });

  it('has correct bone structure', () => {
    const { flameParams } = generateProceduralHead(100);
    expect(flameParams.numBones).toBe(5);
    expect(flameParams.leftEyeBoneIndex).toBe(3);
    expect(flameParams.rightEyeBoneIndex).toBe(4);
  });

  it('assigns LBS weights that sum to 1 per gaussian', () => {
    const { lbsWeights, flameParams } = generateProceduralHead(500);
    for (let i = 0; i < 500; i++) {
      let sum = 0;
      for (let b = 0; b < flameParams.numBones; b++) {
        sum += lbsWeights[i * flameParams.numBones + b];
      }
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });
});

describe('applyEyePose', () => {
  it('returns same positions when gaze is neutral', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(100);
    const neutral: EyePose = {
      left: { pitch: 0, yaw: 0 },
      right: { pitch: 0, yaw: 0 },
    };

    const deformed = applyEyePose(positions, lbsWeights, flameParams, neutral);
    for (let i = 0; i < positions.length; i++) {
      expect(deformed[i]).toBeCloseTo(positions[i], 4);
    }
  });

  it('moves eye gaussians when gaze is non-zero', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(5000);
    const gaze: EyePose = {
      left: { pitch: 0.3, yaw: 0.5 },
      right: { pitch: 0.3, yaw: 0.5 },
    };

    const deformed = applyEyePose(positions, lbsWeights, flameParams, gaze);

    let anyMoved = false;
    for (let i = 0; i < flameParams.numGaussians; i++) {
      const isEye =
        lbsWeights[i * flameParams.numBones + 3] > 0.5 ||
        lbsWeights[i * flameParams.numBones + 4] > 0.5;
      if (isEye) {
        const dx = Math.abs(deformed[i * 3] - positions[i * 3]);
        const dy = Math.abs(deformed[i * 3 + 1] - positions[i * 3 + 1]);
        if (dx > 0.0001 || dy > 0.0001) {
          anyMoved = true;
        }
      }
    }
    expect(anyMoved).toBe(true);
  });

  it('does not move non-eye gaussians', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(5000);
    const gaze: EyePose = {
      left: { pitch: 0.5, yaw: 0.5 },
      right: { pitch: 0.5, yaw: 0.5 },
    };

    const deformed = applyEyePose(positions, lbsWeights, flameParams, gaze);

    for (let i = 0; i < flameParams.numGaussians; i++) {
      const isEye =
        lbsWeights[i * flameParams.numBones + 3] > 0.5 ||
        lbsWeights[i * flameParams.numBones + 4] > 0.5;
      if (!isEye) {
        expect(deformed[i * 3]).toBeCloseTo(positions[i * 3], 4);
        expect(deformed[i * 3 + 1]).toBeCloseTo(positions[i * 3 + 1], 4);
        expect(deformed[i * 3 + 2]).toBeCloseTo(positions[i * 3 + 2], 4);
      }
    }
  });
});
