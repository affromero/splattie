import type { FlameParams, HeadPose } from '@/types/gaze';

export function applyHeadPose(
  canonicalPositions: Float32Array,
  lbsWeights: Float32Array,
  flameParams: FlameParams,
  pose: HeadPose
): Float32Array {
  const numGaussians = flameParams.numGaussians;
  const numBones = flameParams.numBones;
  const deformed = new Float32Array(canonicalPositions.length);

  const neckMat = eyeRotationMatrix(pose.neckYaw, pose.neckPitch);
  const leftMat = eyeRotationMatrix(
    pose.eyes.left.yaw - pose.neckYaw,
    pose.eyes.left.pitch - pose.neckPitch
  );
  const rightMat = eyeRotationMatrix(
    pose.eyes.right.yaw - pose.neckYaw,
    pose.eyes.right.pitch - pose.neckPitch
  );

  const leftCenter = { x: -0.035, y: 0.03, z: 0.07 };
  const rightCenter = { x: 0.035, y: 0.03, z: 0.07 };
  const jawPivot = { x: 0, y: -0.02, z: 0.03 };

  for (let i = 0; i < numGaussians; i++) {
    const cx = canonicalPositions[i * 3];
    const cy = canonicalPositions[i * 3 + 1];
    const cz = canonicalPositions[i * 3 + 2];

    let dx = 0,
      dy = 0,
      dz = 0;

    const wNeck = lbsWeights[i * numBones + flameParams.neckBoneIndex];
    const wJaw = lbsWeights[i * numBones + flameParams.jawBoneIndex];
    const wLeft = lbsWeights[i * numBones + flameParams.leftEyeBoneIndex];
    const wRight = lbsWeights[i * numBones + flameParams.rightEyeBoneIndex];
    const wRoot = lbsWeights[i * numBones + 0];

    if (wNeck > 0.001 || wRoot > 0.001) {
      const w = wNeck + wRoot;
      const [rx, ry, rz] = applyRotation(neckMat, cx, cy, cz);
      dx += w * (rx - cx);
      dy += w * (ry - cy);
      dz += w * (rz - cz);
    }

    if (wJaw > 0.001 && pose.jawOpen > 0.01) {
      const jx = cx - jawPivot.x, jy = cy - jawPivot.y, jz = cz - jawPivot.z;
      const jawAngle = -pose.jawOpen * 0.3;
      const cosA = Math.cos(jawAngle), sinA = Math.sin(jawAngle);
      const ry = jy * cosA - jz * sinA;
      const rz = jy * sinA + jz * cosA;
      dx += wJaw * (jawPivot.x + jx - cx);
      dy += wJaw * (jawPivot.y + ry - cy);
      dz += wJaw * (jawPivot.z + rz - cz);
    }

    if (wLeft > 0.001) {
      const lx = cx - leftCenter.x, ly = cy - leftCenter.y, lz = cz - leftCenter.z;
      const [rx, ry, rz] = applyRotation(leftMat, lx, ly, lz);
      dx += wLeft * (rx - lx);
      dy += wLeft * (ry - ly);
      dz += wLeft * (rz - lz);
    }

    if (wRight > 0.001) {
      const lx = cx - rightCenter.x, ly = cy - rightCenter.y, lz = cz - rightCenter.z;
      const [rx, ry, rz] = applyRotation(rightMat, lx, ly, lz);
      dx += wRight * (rx - lx);
      dy += wRight * (ry - ly);
      dz += wRight * (rz - lz);
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

  const leftEyeCenter = { x: -0.035, y: 0.03, z: 0.07 };
  const rightEyeCenter = { x: 0.035, y: 0.03, z: 0.07 };
  const pupilRadius = 0.006;
  const irisRadius = 0.012;
  const eyeSocketRadius = 0.022;
  const noseCenter = { x: 0, y: -0.01, z: 0.09 };
  const mouthCenter = { x: 0, y: -0.045, z: 0.075 };

  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = Math.cbrt(Math.random());

    const x = r * Math.sin(phi) * Math.cos(theta) * 0.085;
    let y = r * Math.sin(phi) * Math.sin(theta) * 0.11;
    let z = r * Math.cos(phi) * 0.08;

    y += 0.005;

    if (z > 0) {
      const chinFactor = Math.max(0, -y - 0.03) * 3;
      z *= 1 - chinFactor * 0.4;
      const browFactor = Math.max(0, y - 0.06) * 5;
      z *= 1 - browFactor * 0.3;
    }

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    const distLeftEye = dist3(x, y, z, leftEyeCenter.x, leftEyeCenter.y, leftEyeCenter.z);
    const distRightEye = dist3(x, y, z, rightEyeCenter.x, rightEyeCenter.y, rightEyeCenter.z);
    const distNose = dist3(x, y, z, noseCenter.x, noseCenter.y, noseCenter.z);
    const distMouth = Math.abs(y - mouthCenter.y) + Math.abs(z - mouthCenter.z) * 0.5;

    if (distLeftEye < pupilRadius || distRightEye < pupilRadius) {
      colors[i * 4] = 20;
      colors[i * 4 + 1] = 20;
      colors[i * 4 + 2] = 30;
      colors[i * 4 + 3] = 255;
      const isLeft = distLeftEye < distRightEye;
      lbsWeights[i * numBones + (isLeft ? 3 : 4)] = 1.0;
    } else if (distLeftEye < irisRadius || distRightEye < irisRadius) {
      colors[i * 4] = 60;
      colors[i * 4 + 1] = 100;
      colors[i * 4 + 2] = 160;
      colors[i * 4 + 3] = 255;
      const isLeft = distLeftEye < distRightEye;
      lbsWeights[i * numBones + (isLeft ? 3 : 4)] = 0.8;
      lbsWeights[i * numBones + 1] = 0.2;
    } else if (distLeftEye < eyeSocketRadius || distRightEye < eyeSocketRadius) {
      colors[i * 4] = 230;
      colors[i * 4 + 1] = 230;
      colors[i * 4 + 2] = 235;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 1] = 1.0;
    } else if (distNose < 0.02 && z > 0.06) {
      const shade = 190 + Math.floor(Math.random() * 20);
      colors[i * 4] = shade;
      colors[i * 4 + 1] = shade - 25;
      colors[i * 4 + 2] = shade - 40;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 1] = 1.0;
    } else if (distMouth < 0.015 && Math.abs(x) < 0.03 && z > 0.05) {
      colors[i * 4] = 180;
      colors[i * 4 + 1] = 100;
      colors[i * 4 + 2] = 100;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 2] = 1.0;
    } else if (y < -0.03) {
      const shade = 200 + Math.floor(Math.random() * 30);
      colors[i * 4] = shade;
      colors[i * 4 + 1] = shade - 20;
      colors[i * 4 + 2] = shade - 40;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 1] = 0.6;
      lbsWeights[i * numBones + 2] = 0.4;
    } else {
      const shade = 200 + Math.floor(Math.random() * 30);
      colors[i * 4] = shade;
      colors[i * 4 + 1] = shade - 20;
      colors[i * 4 + 2] = shade - 40;
      colors[i * 4 + 3] = 255;
      lbsWeights[i * numBones + 1] = 1.0;
    }
  }

  const flameParams: FlameParams = {
    lbsWeights,
    boneTransforms: new Float32Array(numBones * 16),
    canonicalPositions: new Float32Array(positions),
    numGaussians: count,
    numBones,
    neckBoneIndex: 1,
    jawBoneIndex: 2,
    leftEyeBoneIndex: 3,
    rightEyeBoneIndex: 4,
  };

  return { positions, colors, lbsWeights, flameParams };
}

function eyeRotationMatrix(yaw: number, pitch: number): number[] {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
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

function dist3(
  x1: number, y1: number, z1: number,
  x2: number, y2: number, z2: number,
): number {
  return Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2);
}
