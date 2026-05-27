import * as THREE from 'three';
import type { CameraConfig } from '../types';

export class CameraSphere {
  private lookAtPoint = new THREE.Vector3(0, 0, 0);

  setLookAt(point: THREE.Vector3 | [number, number, number]): void {
    if (Array.isArray(point)) {
      this.lookAtPoint.set(point[0], point[1], point[2]);
    } else {
      this.lookAtPoint.copy(point);
    }
  }

  apply(camera: THREE.PerspectiveCamera, config: CameraConfig): void {
    const thetaRad = (config.theta * Math.PI) / 180;
    const phiRad = (config.phi * Math.PI) / 180;
    const r = config.radius;

    camera.position.set(
      this.lookAtPoint.x + r * Math.sin(phiRad) * Math.sin(thetaRad),
      this.lookAtPoint.y + r * Math.cos(phiRad),
      this.lookAtPoint.z + r * Math.sin(phiRad) * Math.cos(thetaRad),
    );

    camera.lookAt(this.lookAtPoint);

    if (config.fov && camera.fov !== config.fov) {
      camera.fov = config.fov;
      camera.updateProjectionMatrix();
    }
  }
}
