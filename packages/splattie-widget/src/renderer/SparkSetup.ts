import * as THREE from 'three';

export interface BoneInfo {
  name: string;
  pos: [number, number, number];
  idx: number;
  parentIdx: number;
  virtual?: boolean;
}

function computeVirtualBones(bones: BoneInfo[]): BoneInfo[] {
  if (bones.length < 5) return [];
  const jaw = bones[2].pos;
  const lEye = bones[3].pos;
  const rEye = bones[4].pos;
  const midX = (lEye[0] + rEye[0]) / 2;
  const midY = (lEye[1] + rEye[1]) / 2;
  const midZ = (lEye[2] + rEye[2]) / 2;
  const eyeSep = Math.abs(lEye[0] - rEye[0]);

  const mouthY = jaw[1] + (midY - jaw[1]) * 0.15;
  const mouthZ = jaw[2] + (midZ - jaw[2]) * 0.17;
  const cheekY = jaw[1] + (midY - jaw[1]) * 0.35;
  const cheekZ = jaw[2] + (midZ - jaw[2]) * 0.25;
  const noseY = jaw[1] + (midY - jaw[1]) * 0.55;
  const noseZ = jaw[2] + (midZ - jaw[2]) * 0.45;

  const idx = bones.length;
  return [
    { name: 'browL', pos: [lEye[0], lEye[1] + 0.015, lEye[2]], idx: idx, parentIdx: 1, virtual: true },
    { name: 'browR', pos: [rEye[0], rEye[1] + 0.015, rEye[2]], idx: idx + 1, parentIdx: 1, virtual: true },
    { name: 'mouthCornerL', pos: [midX + eyeSep * 0.3, mouthY, mouthZ], idx: idx + 2, parentIdx: 2, virtual: true },
    { name: 'mouthCornerR', pos: [midX - eyeSep * 0.3, mouthY, mouthZ], idx: idx + 3, parentIdx: 2, virtual: true },
    { name: 'cheekL', pos: [lEye[0] + 0.01, cheekY, cheekZ], idx: idx + 4, parentIdx: 1, virtual: true },
    { name: 'cheekR', pos: [rEye[0] - 0.01, cheekY, cheekZ], idx: idx + 5, parentIdx: 1, virtual: true },
    { name: 'noseBridge', pos: [midX, noseY, noseZ], idx: idx + 6, parentIdx: 1, virtual: true },
  ];
}

async function parsePlyPositions(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  const buffer = await res.arrayBuffer();
  const headerBytes = new Uint8Array(buffer, 0, Math.min(4096, buffer.byteLength));
  const header = new TextDecoder().decode(headerBytes);
  const endIdx = header.indexOf('end_header');
  const headerEnd = buffer.byteLength > endIdx ? header.indexOf('\n', endIdx) + 1 : 0;

  const vertexMatch = header.match(/element vertex (\d+)/);
  const numVerts = vertexMatch ? parseInt(vertexMatch[1]) : 0;

  const props: string[] = [];
  for (const line of header.split('\n')) {
    if (line.startsWith('property ')) props.push(line.trim());
    if (line === 'end_header') break;
  }

  let bytesPerVertex = 0;
  for (const p of props) {
    if (p.includes('float')) bytesPerVertex += 4;
    else if (p.includes('double')) bytesPerVertex += 8;
    else if (p.includes('uchar') || p.includes('uint8')) bytesPerVertex += 1;
    else if (p.includes('short') || p.includes('int16')) bytesPerVertex += 2;
    else if (p.includes('int') || p.includes('uint')) bytesPerVertex += 4;
  }

  const data = new DataView(buffer, headerEnd);
  const positions = new Float32Array(numVerts * 3);
  for (let i = 0; i < numVerts; i++) {
    const offset = i * bytesPerVertex;
    positions[i * 3] = data.getFloat32(offset, true);
    positions[i * 3 + 1] = data.getFloat32(offset + 4, true);
    positions[i * 3 + 2] = data.getFloat32(offset + 8, true);
  }
  return positions;
}

export interface SparkInstance {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  splatMesh: unknown;
  skinning: unknown;
  bones: BoneInfo[];
  canvas: HTMLCanvasElement;
  baselinePositions: Float32Array | null;
  packedArray: Uint32Array | null;
  packedSplatsRef: { needsUpdate: boolean } | null;
}

export async function createSparkInstance(
  container: HTMLElement,
  splatUrl: string,
  backgroundColor: number,
  boneTreeUrl?: string,
  lbsWeightsUrl?: string,
): Promise<SparkInstance> {
  const { SparkRenderer, SplatMesh, SplatSkinning, SplatSkinningMode } = await import('@sparkjsdev/spark');

  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 500;

  const renderer = new THREE.WebGLRenderer({ canvas });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(backgroundColor);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.001, 100);
  camera.position.set(0, 0, 0.5);
  camera.lookAt(0, 0, 0);

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const splatMesh = await new Promise<InstanceType<typeof SplatMesh>>((resolve) => {
    const mesh = new SplatMesh({
      url: splatUrl,
      sphericalHarmonicsDegree: 0,
      onLoad: () => {
        // Auto-center camera on the loaded splat
        const box = new THREE.Box3().setFromObject(mesh as unknown as THREE.Object3D);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 0) {
          camera.position.set(center.x, center.y, center.z + maxDim * 2);
          camera.lookAt(center);
        }
        resolve(mesh);
      },
    } as Record<string, unknown>);
    scene.add(mesh);
  });

  let skinning: InstanceType<typeof SplatSkinning> | null = null;
  const bones: BoneInfo[] = [];
  let baselinePositions: Float32Array | null = null;

  if (boneTreeUrl && lbsWeightsUrl) {
    const [boneTree, lbsWeights, splatPositions] = await Promise.all([
      fetch(boneTreeUrl).then((r) => r.json()),
      fetch(lbsWeightsUrl).then((r) => r.json()),
      parsePlyPositions(splatUrl),
    ]) as [{ bones: Array<{ name: string; position: number[]; children?: unknown[] }> }, number[][], Float32Array];

    function flattenBones(
      node: { name: string; position: number[]; children?: unknown[] },
      parentIdx: number,
    ): void {
      const idx = bones.length;
      bones.push({ name: node.name, pos: node.position as [number, number, number], idx, parentIdx });
      const children = node.children as Array<{ name: string; position: number[]; children?: unknown[] }> | undefined;
      if (children) children.forEach((c) => flattenBones(c, idx));
    }
    flattenBones(boneTree.bones[0], -1);

    const virtualBones = computeVirtualBones(bones);
    bones.push(...virtualBones);

    skinning = new SplatSkinning({
      mesh: splatMesh,
      numBones: bones.length,
      mode: SplatSkinningMode.DUAL_QUATERNION,
    });

    const identityQuat = new THREE.Quaternion();
    for (const bone of bones) {
      skinning.setRestQuatPos(bone.idx, identityQuat, new THREE.Vector3(...bone.pos));
    }

    const numSplats = Math.min(lbsWeights.length, skinning.numSplats);
    const mouthBoneNames = new Set(['mouthCornerL', 'mouthCornerR']);
    type Sigma3 = [number, number, number];
    const sigmaByBone = new Map<number, Sigma3>();
    for (const vb of virtualBones) {
      sigmaByBone.set(vb.idx, mouthBoneNames.has(vb.name)
        ? [0.035, 0.02, 0.035]
        : [0.02, 0.02, 0.02]);
    }

    for (let i = 0; i < numSplats; i++) {
      const origWeights = lbsWeights[i];
      const allWeights: [number, number][] = origWeights.map((val, idx) => [idx, val]);

      if (virtualBones.length > 0 && i * 3 + 2 < splatPositions.length) {
        const px = splatPositions[i * 3];
        const py = splatPositions[i * 3 + 1];
        const pz = splatPositions[i * 3 + 2];
        const jawW = origWeights[2] ?? 0;
        const eyeLW = origWeights[3] ?? 0;
        const eyeRW = origWeights[4] ?? 0;
        const eyeW = Math.max(eyeLW, eyeRW);

        for (const vb of virtualBones) {
          if (vb.name.startsWith('brow') && (eyeW > 0.12 || py < vb.pos[1] - 0.005)) continue;
          if (vb.name.startsWith('mouth') && jawW < 0.15) continue;
          if (vb.name.startsWith('cheek') && jawW < 0.1 && eyeW > 0.3) continue;
          if (vb.name === 'noseBridge' && eyeW > 0.4) continue;

          const dx = px - vb.pos[0], dy = py - vb.pos[1], dz = pz - vb.pos[2];
          const [sx, sy, sz] = sigmaByBone.get(vb.idx) ?? [0.02, 0.02, 0.02];
          const nd2 = (dx * dx) / (2 * sx * sx) + (dy * dy) / (2 * sy * sy) + (dz * dz) / (2 * sz * sz);
          const w = Math.exp(-nd2) * 3.0;
          if (w > 0.01) allWeights.push([vb.idx, w]);
        }
      }

      allWeights.sort((a, b) => b[1] - a[1]);
      const top4 = allWeights.slice(0, 4);
      const sum = top4.reduce((s, p) => s + p[1], 0) || 1;
      skinning.setSplatBones(
        i,
        new THREE.Vector4(top4[0]?.[0] ?? 0, top4[1]?.[0] ?? 0, top4[2]?.[0] ?? 0, top4[3]?.[0] ?? 0),
        new THREE.Vector4((top4[0]?.[1] ?? 0) / sum, (top4[1]?.[1] ?? 0) / sum, (top4[2]?.[1] ?? 0) / sum, (top4[3]?.[1] ?? 0) / sum),
      );
    }

    (splatMesh as unknown as { skinning: unknown }).skinning = skinning;
    skinning.updateBones();
    baselinePositions = splatPositions;
  }

  // Expose packed splat buffer for expression basis per-splat position updates
  let packedArray: Uint32Array | null = null;
  let packedSplatsRef: { needsUpdate: boolean } | null = null;

  const meshAny = splatMesh as Record<string, unknown>;
  const ps = meshAny.packedSplats as { packedArray?: Uint32Array; needsUpdate?: boolean } | undefined;
  if (ps?.packedArray) {
    packedArray = ps.packedArray;
    packedSplatsRef = ps as { needsUpdate: boolean };
  }

  return { renderer, scene, camera, splatMesh, skinning, bones, canvas, baselinePositions, packedArray, packedSplatsRef };
}
