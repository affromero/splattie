import type { EyePose, FlameParams } from '@/types/gaze';

export function applyEyePose(
  canonicalPositions: Float32Array,
  lbsWeights: Float32Array,
  flameParams: FlameParams,
  eyePose: EyePose
): Float32Array {
  const numGaussians = flameParams.numGaussians;
  const numBones = flameParams.numBones;
  const deformed = new Float32Array(canonicalPositions.length);

  const leftMat = eyeRotationMatrix(eyePose.left.yaw, eyePose.left.pitch);
  const rightMat = eyeRotationMatrix(eyePose.right.yaw, eyePose.right.pitch);

  for (let i = 0; i < numGaussians; i++) {
    const cx = canonicalPositions[i * 3];
    const cy = canonicalPositions[i * 3 + 1];
    const cz = canonicalPositions[i * 3 + 2];

    let dx = 0,
      dy = 0,
      dz = 0;

    for (let b = 0; b < numBones; b++) {
      const w = lbsWeights[i * numBones + b];
      if (w < 0.001) continue;

      if (b === flameParams.leftEyeBoneIndex) {
        const [rx, ry, rz] = applyRotation(leftMat, cx, cy, cz);
        dx += w * (rx - cx);
        dy += w * (ry - cy);
        dz += w * (rz - cz);
      } else if (b === flameParams.rightEyeBoneIndex) {
        const [rx, ry, rz] = applyRotation(rightMat, cx, cy, cz);
        dx += w * (rx - cx);
        dy += w * (ry - cy);
        dz += w * (rz - cz);
      }
    }

    deformed[i * 3] = cx + dx;
    deformed[i * 3 + 1] = cy + dy;
    deformed[i * 3 + 2] = cz + dz;
  }

  return deformed;
}

export function generateProceduralHead(count: number): {
  positions: Float32Array;
  colors: Uint8Array;
  lbsWeights: Float32Array;
  flameParams: FlameParams;
} {
  const positions = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 4);
  const numBones = 5;
  const lbsWeights = new Float32Array(count * numBones);

  const leftEyeCenter = { x: -0.03, y: 0.02, z: 0.08 };
  const rightEyeCenter = { x: 0.03, y: 0.02, z: 0.08 };
  const eyeRadius = 0.015;

  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 0.1 * Math.cbrt(Math.random());

    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta) * 1.2;
    const z = r * Math.cos(phi) * 0.9;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    const distLeft = Math.sqrt(
      (x - leftEyeCenter.x) ** 2 +
        (y - leftEyeCenter.y) ** 2 +
        (z - leftEyeCenter.z) ** 2
    );
    const distRight = Math.sqrt(
      (x - rightEyeCenter.x) ** 2 +
        (y - rightEyeCenter.y) ** 2 +
        (z - rightEyeCenter.z) ** 2
    );

    const isLeftEye = distLeft < eyeRadius;
    const isRightEye = distRight < eyeRadius;

    if (isLeftEye) {
      colors[i * 4] = 40;
      colors[i * 4 + 1] = 40;
      colors[i * 4 + 2] = 60;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 3] = 1.0;
    } else if (isRightEye) {
      colors[i * 4] = 40;
      colors[i * 4 + 1] = 40;
      colors[i * 4 + 2] = 60;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 4] = 1.0;
    } else {
      const skinTone = 180 + Math.floor(Math.random() * 40);
      colors[i * 4] = skinTone;
      colors[i * 4 + 1] = skinTone - 30;
      colors[i * 4 + 2] = skinTone - 50;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 0] = 1.0;
    }
  }

  const flameParams: FlameParams = {
    lbsWeights,
    boneTransforms: new Float32Array(numBones * 16),
    canonicalPositions: positions,
    numGaussians: count,
    numBones,
    leftEyeBoneIndex: 3,
    rightEyeBoneIndex: 4,
  };

  return { positions, colors, lbsWeights, flameParams };
}

function eyeRotationMatrix(yaw: number, pitch: number): number[] {
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw);
  const cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  return [
    cy, 0, sy,
    sy * sp, cp, -cy * sp,
    -sy * cp, sp, cy * cp,
  ];
}

function applyRotation(m: number[], x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
}
