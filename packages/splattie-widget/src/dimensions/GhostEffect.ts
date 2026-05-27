import type { GhostConfig } from '../types';

export class GhostEffect {
  apply(
    mesh: { position: { y: number }; rotation: { x: number; z: number } },
    ghost: GhostConfig,
    time: number,
  ): void {
    const phase = ghost.phase ?? 0;
    mesh.position.y += Math.sin(time * ghost.frequency * Math.PI * 2 + phase) * ghost.amplitude;

    if (ghost.wobble > 0) {
      mesh.rotation.x += Math.sin(time * 0.7 + phase) * ghost.wobble * 0.01;
      mesh.rotation.z += Math.cos(time * 0.5 + phase) * ghost.wobble * 0.008;
    }
  }
}
