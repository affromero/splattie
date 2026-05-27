import { describe, expect, it } from 'vitest';
import {
  getEasing,
  interpolateTransition,
  lerpCamera,
  lerpExpression,
  lerpGhost,
  lerpNumber,
  lerpRotation,
  lerpState,
  lerpTracking,
} from '../src/state/Interpolator';
import type { StateDefinition } from '../src/types';

describe('lerpNumber', () => {
  it('interpolates between two numbers', () => {
    expect(lerpNumber(0, 10, 0.5)).toBe(5);
    expect(lerpNumber(0, 10, 0)).toBe(0);
    expect(lerpNumber(0, 10, 1)).toBe(10);
  });
});

describe('lerpGhost', () => {
  it('interpolates ghost config', () => {
    const a = { amplitude: 0, frequency: 0, wobble: 0 };
    const b = { amplitude: 1, frequency: 2, wobble: 0.5 };
    const result = lerpGhost(a, b, 0.5);
    expect(result.amplitude).toBeCloseTo(0.5);
    expect(result.frequency).toBeCloseTo(1);
    expect(result.wobble).toBeCloseTo(0.25);
  });
});

describe('lerpExpression', () => {
  it('interpolates shared keys', () => {
    const result = lerpExpression({ jawOpen: 0 }, { jawOpen: 1 }, 0.5);
    expect(result.jawOpen).toBeCloseTo(0.5);
  });

  it('handles missing keys as 0', () => {
    const result = lerpExpression({}, { jawOpen: 1 }, 0.5);
    expect(result.jawOpen).toBeCloseTo(0.5);
  });
});

describe('lerpCamera', () => {
  it('interpolates camera on sphere', () => {
    const a = { theta: 0, phi: 90, radius: 3, fov: 50 };
    const b = { theta: 30, phi: 60, radius: 2, fov: 40 };
    const result = lerpCamera(a, b, 0.5);
    expect(result.theta).toBeCloseTo(15);
    expect(result.phi).toBeCloseTo(75);
    expect(result.radius).toBeCloseTo(2.5);
    expect(result.fov).toBeCloseTo(45);
  });
});

describe('lerpRotation', () => {
  it('interpolates rotation', () => {
    const result = lerpRotation([0, 0, 0], [10, 20, 30], 0.5);
    expect(result).toEqual([5, 10, 15]);
  });
});

describe('lerpTracking', () => {
  it('interpolates tracking intensities', () => {
    const result = lerpTracking({ eyes: 1, head: 0 }, { eyes: 0, head: 1 }, 0.5);
    expect(result.eyes).toBeCloseTo(0.5);
    expect(result.head).toBeCloseTo(0.5);
  });
});

describe('getEasing', () => {
  it('returns linear for unknown', () => {
    expect(getEasing('unknown')(0.5)).toBe(0.5);
  });

  it('ease-out is faster at start', () => {
    expect(getEasing('ease-out')(0.5)).toBeGreaterThan(0.5);
  });

  it('snap returns 1 immediately', () => {
    expect(getEasing('snap')(0)).toBe(1);
    expect(getEasing('snap')(0.5)).toBe(1);
  });
});

describe('interpolateTransition', () => {
  const stateA: StateDefinition = {
    ghost: { amplitude: 0, frequency: 0, wobble: 0 },
    expression: {},
    camera: { theta: 0, phi: 90, radius: 3 },
    rotation: [0, 0, 0],
    tracking: { eyes: 1, head: 0 },
  };

  const stateB: StateDefinition = {
    ghost: { amplitude: 1, frequency: 1, wobble: 1 },
    expression: { jawOpen: 1 },
    camera: { theta: 30, phi: 60, radius: 2 },
    rotation: [10, 20, 30],
    tracking: { eyes: 0, head: 1 },
  };

  it('returns halfway state at half duration', () => {
    const { state, done } = interpolateTransition(stateA, stateB, 0.15, {
      duration: 0.3,
      easing: 'linear',
    });
    expect(done).toBe(false);
    expect(state.ghost.amplitude).toBeCloseTo(0.5);
    expect(state.camera.theta).toBeCloseTo(15);
  });

  it('returns target state when done', () => {
    const { state, done } = interpolateTransition(stateA, stateB, 0.5, {
      duration: 0.3,
      easing: 'linear',
    });
    expect(done).toBe(true);
    expect(state.ghost.amplitude).toBeCloseTo(1);
  });

  it('snap returns target immediately', () => {
    const { state, done } = interpolateTransition(stateA, stateB, 0, {
      duration: 0.3,
      easing: 'snap',
    });
    expect(done).toBe(true);
    expect(state.ghost.amplitude).toBe(1);
  });
});
