import { describe, expect, it } from 'vitest';
import { lerpEyePose, lerpGaze, lerpHeadPose, mouseToNDC, screenToGaze, screenToHeadPose } from '../gaze-math';

describe('mouseToNDC', () => {
  const rect = { left: 0, top: 0, width: 800, height: 600 } as DOMRect;

  it('converts center to (0, 0)', () => {
    const ndc = mouseToNDC(400, 300, rect);
    expect(ndc.x).toBeCloseTo(0);
    expect(ndc.y).toBeCloseTo(0);
  });

  it('converts top-left to (-1, -1)', () => {
    const ndc = mouseToNDC(0, 0, rect);
    expect(ndc.x).toBeCloseTo(-1);
    expect(ndc.y).toBeCloseTo(-1);
  });

  it('converts bottom-right to (1, 1)', () => {
    const ndc = mouseToNDC(800, 600, rect);
    expect(ndc.x).toBeCloseTo(1);
    expect(ndc.y).toBeCloseTo(1);
  });
});

describe('screenToGaze', () => {
  it('returns neutral gaze at center', () => {
    const pose = screenToGaze({ x: 0, y: 0 });
    expect(pose.left.pitch).toBeCloseTo(0);
    expect(pose.left.yaw).toBeCloseTo(0);
  });

  it('clamps at extremes', () => {
    const maxYaw = Math.PI / 4;
    const maxPitch = Math.PI / 6;
    const pose = screenToGaze({ x: 10, y: 10 });
    expect(pose.left.yaw).toBeCloseTo(maxYaw);
    expect(pose.left.pitch).toBeCloseTo(maxPitch);
  });

  it('both eyes get the same gaze', () => {
    const pose = screenToGaze({ x: 0.5, y: -0.3 });
    expect(pose.left.yaw).toEqual(pose.right.yaw);
    expect(pose.left.pitch).toEqual(pose.right.pitch);
  });
});

describe('screenToHeadPose', () => {
  it('includes neck rotation following gaze', () => {
    const pose = screenToHeadPose({ x: 0.5, y: 0 });
    expect(pose.neckYaw).not.toBe(0);
    expect(Math.abs(pose.neckYaw)).toBeLessThan(Math.abs(pose.eyes.left.yaw));
  });

  it('opens jaw when cursor is at center (on face)', () => {
    const pose = screenToHeadPose({ x: 0, y: 0 });
    expect(pose.jawOpen).toBeGreaterThan(0);
  });

  it('closes jaw when cursor is far from center', () => {
    const pose = screenToHeadPose({ x: 0.9, y: 0.9 });
    expect(pose.jawOpen).toBe(0);
  });
});

describe('lerpGaze', () => {
  it('interpolates between two gaze vectors', () => {
    const a = { pitch: 0, yaw: 0 };
    const b = { pitch: 1, yaw: 1 };
    const result = lerpGaze(a, b, 0.5);
    expect(result.pitch).toBeCloseTo(0.5);
    expect(result.yaw).toBeCloseTo(0.5);
  });

  it('returns start at t=0', () => {
    const a = { pitch: 0.5, yaw: 0.3 };
    const b = { pitch: 1, yaw: 1 };
    const result = lerpGaze(a, b, 0);
    expect(result.pitch).toBeCloseTo(0.5);
    expect(result.yaw).toBeCloseTo(0.3);
  });
});

describe('lerpEyePose', () => {
  it('interpolates both eyes', () => {
    const a = { left: { pitch: 0, yaw: 0 }, right: { pitch: 0, yaw: 0 } };
    const b = { left: { pitch: 1, yaw: 1 }, right: { pitch: -1, yaw: -1 } };
    const result = lerpEyePose(a, b, 0.5);
    expect(result.left.pitch).toBeCloseTo(0.5);
    expect(result.right.yaw).toBeCloseTo(-0.5);
  });
});

describe('lerpHeadPose', () => {
  it('interpolates all channels', () => {
    const a = { eyes: { left: { pitch: 0, yaw: 0 }, right: { pitch: 0, yaw: 0 } }, neckYaw: 0, neckPitch: 0, jawOpen: 0 };
    const b = { eyes: { left: { pitch: 1, yaw: 1 }, right: { pitch: 1, yaw: 1 } }, neckYaw: 0.5, neckPitch: 0.3, jawOpen: 0.8 };
    const result = lerpHeadPose(a, b, 0.5);
    expect(result.neckYaw).toBeCloseTo(0.25);
    expect(result.neckPitch).toBeCloseTo(0.15);
    expect(result.jawOpen).toBeGreaterThan(0);
  });
});
