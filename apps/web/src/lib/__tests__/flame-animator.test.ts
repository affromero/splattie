import { describe, expect, it } from 'vitest';
import { applyHeadPose, generateProceduralHead } from '../flame-animator';
import type { HeadPose } from '@/types/gaze';

const NEUTRAL: HeadPose = {
  eyes: { left: { pitch: 0, yaw: 0 }, right: { pitch: 0, yaw: 0 } },
  neckYaw: 0,
  neckPitch: 0,
  jawOpen: 0,
};

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
    expect(flameParams.neckBoneIndex).toBe(1);
    expect(flameParams.jawBoneIndex).toBe(2);
    expect(flameParams.leftEyeBoneIndex).toBe(3);
    expect(flameParams.rightEyeBoneIndex).toBe(4);
  });

  it('assigns LBS weights that sum to ~1 per gaussian', () => {
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

describe('applyHeadPose', () => {
  it('returns same positions when pose is neutral', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(100);
    const deformed = applyHeadPose(positions, lbsWeights, flameParams, NEUTRAL);
    for (let i = 0; i < positions.length; i++) {
      expect(deformed[i]).toBeCloseTo(positions[i], 4);
    }
  });

  it('moves head when neck rotates', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(5000);
    const pose: HeadPose = { ...NEUTRAL, neckYaw: 0.3, neckPitch: 0.2 };

    const deformed = applyHeadPose(positions, lbsWeights, flameParams, pose);

    let totalDisplacement = 0;
    for (let i = 0; i < flameParams.numGaussians; i++) {
      totalDisplacement += Math.abs(deformed[i * 3] - positions[i * 3]);
    }
    expect(totalDisplacement).toBeGreaterThan(0);
  });

  it('opens jaw when jawOpen > 0', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(5000);
    const pose: HeadPose = { ...NEUTRAL, jawOpen: 0.5 };

    const deformed = applyHeadPose(positions, lbsWeights, flameParams, pose);

    let jawMoved = false;
    for (let i = 0; i < flameParams.numGaussians; i++) {
      const wJaw = lbsWeights[i * flameParams.numBones + flameParams.jawBoneIndex];
      if (wJaw > 0.5) {
        const dy = Math.abs(deformed[i * 3 + 1] - positions[i * 3 + 1]);
        if (dy > 0.0001) jawMoved = true;
      }
    }
    expect(jawMoved).toBe(true);
  });

  it('moves eye gaussians when gaze is non-zero', () => {
    const { positions, lbsWeights, flameParams } = generateProceduralHead(5000);
    const pose: HeadPose = {
      ...NEUTRAL,
      eyes: { left: { pitch: 0.3, yaw: 0.5 }, right: { pitch: 0.3, yaw: 0.5 } },
    };

    const deformed = applyHeadPose(positions, lbsWeights, flameParams, pose);

    let anyMoved = false;
    for (let i = 0; i < flameParams.numGaussians; i++) {
      const wLeft = lbsWeights[i * flameParams.numBones + 3];
      const wRight = lbsWeights[i * flameParams.numBones + 4];
      if (wLeft > 0.5 || wRight > 0.5) {
        const dx = Math.abs(deformed[i * 3] - positions[i * 3]);
        if (dx > 0.0001) anyMoved = true;
      }
    }
    expect(anyMoved).toBe(true);
  });
});
