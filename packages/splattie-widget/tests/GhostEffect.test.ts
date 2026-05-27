import { describe, expect, it } from 'vitest';
import { GhostEffect } from '../src/dimensions/GhostEffect';

describe('GhostEffect', () => {
  it('offsets mesh position by amplitude', () => {
    const ghost = new GhostEffect();
    const mesh = { position: { y: 0 }, rotation: { x: 0, z: 0 } };
    ghost.apply(mesh, { amplitude: 0.01, frequency: 1, wobble: 0 }, 0.25);
    expect(mesh.position.y).not.toBe(0);
    expect(Math.abs(mesh.position.y)).toBeLessThanOrEqual(0.01);
  });

  it('does not wobble when wobble is 0', () => {
    const ghost = new GhostEffect();
    const mesh = { position: { y: 0 }, rotation: { x: 0, z: 0 } };
    ghost.apply(mesh, { amplitude: 0.01, frequency: 1, wobble: 0 }, 1);
    expect(mesh.rotation.x).toBe(0);
    expect(mesh.rotation.z).toBe(0);
  });

  it('wobbles when wobble > 0', () => {
    const ghost = new GhostEffect();
    const mesh = { position: { y: 0 }, rotation: { x: 0, z: 0 } };
    ghost.apply(mesh, { amplitude: 0.01, frequency: 1, wobble: 1.0 }, 1);
    expect(mesh.rotation.x).not.toBe(0);
    expect(mesh.rotation.z).not.toBe(0);
  });

  it('respects phase offset', () => {
    const ghost = new GhostEffect();
    const mesh1 = { position: { y: 0 }, rotation: { x: 0, z: 0 } };
    const mesh2 = { position: { y: 0 }, rotation: { x: 0, z: 0 } };
    ghost.apply(mesh1, { amplitude: 0.01, frequency: 1, wobble: 0, phase: 0 }, 0.1);
    ghost.apply(mesh2, { amplitude: 0.01, frequency: 1, wobble: 0, phase: Math.PI / 2 }, 0.1);
    expect(Math.abs(mesh1.position.y - mesh2.position.y)).toBeGreaterThan(0.001);
  });
});
