import type { CameraConfig, GhostConfig, StateDefinition, TrackingConfig, TransitionConfig } from '../types';

type EasingFn = (t: number) => number;

const EASINGS: Record<string, EasingFn> = {
  linear: (t) => t,
  'ease-in': (t) => t * t,
  'ease-out': (t) => t * (2 - t),
  'ease-in-out': (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  snap: () => 1,
};

export function getEasing(name: string): EasingFn {
  return EASINGS[name] ?? EASINGS.linear;
}

export function lerpNumber(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function lerpGhost(a: GhostConfig, b: GhostConfig, t: number): GhostConfig {
  return {
    amplitude: lerpNumber(a.amplitude, b.amplitude, t),
    frequency: lerpNumber(a.frequency, b.frequency, t),
    wobble: lerpNumber(a.wobble, b.wobble, t),
    phase: lerpNumber(a.phase ?? 0, b.phase ?? 0, t),
  };
}

export function lerpExpression(
  a: Record<string, number>,
  b: Record<string, number>,
  t: number,
): Record<string, number> {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const result: Record<string, number> = {};
  for (const k of keys) {
    result[k] = lerpNumber(a[k] ?? 0, b[k] ?? 0, t);
  }
  return result;
}

export function lerpCamera(a: CameraConfig, b: CameraConfig, t: number): CameraConfig {
  return {
    theta: lerpNumber(a.theta, b.theta, t),
    phi: lerpNumber(a.phi, b.phi, t),
    radius: lerpNumber(a.radius, b.radius, t),
    lookAt: a.lookAt,
    fov: lerpNumber(a.fov ?? 50, b.fov ?? 50, t),
  };
}

export function lerpRotation(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  return [lerpNumber(a[0], b[0], t), lerpNumber(a[1], b[1], t), lerpNumber(a[2], b[2], t)];
}

export function lerpTracking(a: TrackingConfig, b: TrackingConfig, t: number): TrackingConfig {
  return {
    eyes: lerpNumber(a.eyes, b.eyes, t),
    head: lerpNumber(a.head, b.head, t),
    body: lerpNumber(a.body ?? 0, b.body ?? 0, t),
  };
}

export function lerpState(
  from: StateDefinition,
  to: StateDefinition,
  t: number,
): StateDefinition {
  return {
    ghost: lerpGhost(from.ghost, to.ghost, t),
    expression: lerpExpression(from.expression, to.expression, t),
    camera: lerpCamera(from.camera, to.camera, t),
    rotation: lerpRotation(from.rotation, to.rotation, t),
    tracking: lerpTracking(from.tracking, to.tracking, t),
  };
}

export function interpolateTransition(
  from: StateDefinition,
  to: StateDefinition,
  elapsed: number,
  transition: TransitionConfig,
): { state: StateDefinition; done: boolean } {
  if (transition.easing === 'snap') {
    return { state: to, done: true };
  }
  const rawT = Math.min(elapsed / transition.duration, 1);
  const easedT = getEasing(transition.easing)(rawT);
  return {
    state: lerpState(from, to, easedT),
    done: rawT >= 1,
  };
}
