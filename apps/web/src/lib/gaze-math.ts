import type { EyePose, GazeVector } from '@/types/gaze';

const MAX_PITCH = Math.PI / 6;
const MAX_YAW = Math.PI / 4;
const SMOOTHING = 0.12;

export function screenToGaze(mouseNDC: { x: number; y: number }): EyePose {
  const yaw = clamp(mouseNDC.x * MAX_YAW, -MAX_YAW, MAX_YAW);
  const pitch = clamp(-mouseNDC.y * MAX_PITCH, -MAX_PITCH, MAX_PITCH);

  return {
    left: { pitch, yaw },
    right: { pitch, yaw },
  };
}

export function lerpGaze(current: GazeVector, target: GazeVector, t: number): GazeVector {
  return {
    pitch: lerp(current.pitch, target.pitch, t),
    yaw: lerp(current.yaw, target.yaw, t),
  };
}

export function lerpEyePose(current: EyePose, target: EyePose, t: number = SMOOTHING): EyePose {
  return {
    left: lerpGaze(current.left, target.left, t),
    right: lerpGaze(current.right, target.right, t),
  };
}

export function mouseToNDC(
  clientX: number,
  clientY: number,
  rect: DOMRect
): { x: number; y: number } {
  return {
    x: ((clientX - rect.left) / rect.width) * 2 - 1,
    y: ((clientY - rect.top) / rect.height) * 2 - 1,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
