import type { EyePose, GazeVector, HeadPose } from '@/types/gaze';

const MAX_PITCH = Math.PI / 6;
const MAX_YAW = Math.PI / 4;
const HEAD_FOLLOW_RATIO = 0.3;
const SMOOTHING = 0.12;
const FACE_RADIUS = 0.35;

export function screenToHeadPose(mouseNDC: { x: number; y: number }): HeadPose {
  const yaw = clamp(mouseNDC.x * MAX_YAW, -MAX_YAW, MAX_YAW);
  const pitch = clamp(-mouseNDC.y * MAX_PITCH, -MAX_PITCH, MAX_PITCH);

  const cursorOnFace =
    mouseNDC.x * mouseNDC.x + mouseNDC.y * mouseNDC.y < FACE_RADIUS * FACE_RADIUS;

  return {
    eyes: {
      left: { pitch, yaw },
      right: { pitch, yaw },
    },
    neckYaw: yaw * HEAD_FOLLOW_RATIO,
    neckPitch: pitch * HEAD_FOLLOW_RATIO,
    jawOpen: cursorOnFace ? 0.4 + Math.abs(mouseNDC.y) * 0.3 : 0,
  };
}

export function screenToGaze(mouseNDC: { x: number; y: number }): EyePose {
  return screenToHeadPose(mouseNDC).eyes;
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

export function lerpHeadPose(
  current: HeadPose,
  target: HeadPose,
  t: number = SMOOTHING
): HeadPose {
  return {
    eyes: lerpEyePose(current.eyes, target.eyes, t),
    neckYaw: lerp(current.neckYaw, target.neckYaw, t),
    neckPitch: lerp(current.neckPitch, target.neckPitch, t),
    jawOpen: lerp(current.jawOpen, target.jawOpen, t * 0.5),
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
