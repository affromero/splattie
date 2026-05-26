export interface GazeVector {
  pitch: number;
  yaw: number;
}

export interface EyePose {
  left: GazeVector;
  right: GazeVector;
}

export interface HeadPose {
  eyes: EyePose;
  neckYaw: number;
  neckPitch: number;
  jawOpen: number;
}

export interface FlameParams {
  lbsWeights: Float32Array;
  boneTransforms: Float32Array;
  canonicalPositions: Float32Array;
  numGaussians: number;
  numBones: number;
  neckBoneIndex: number;
  jawBoneIndex: number;
  leftEyeBoneIndex: number;
  rightEyeBoneIndex: number;
}
