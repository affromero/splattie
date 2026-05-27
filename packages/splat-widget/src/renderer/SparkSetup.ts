import * as THREE from 'three';

export interface BoneInfo {
  name: string;
  pos: [number, number, number];
  idx: number;
  parentIdx: number;
}

export interface SparkInstance {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  splatMesh: unknown;
  skinning: unknown;
  bones: BoneInfo[];
  canvas: HTMLCanvasElement;
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

  if (boneTreeUrl && lbsWeightsUrl) {
    const [boneTree, lbsWeights] = await Promise.all([
      fetch(boneTreeUrl).then((r) => r.json()),
      fetch(lbsWeightsUrl).then((r) => r.json()),
    ]) as [{ bones: Array<{ name: string; position: number[]; children?: unknown[] }> }, number[][]];

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

    skinning = new SplatSkinning({
      mesh: splatMesh,
      numBones: bones.length,
      mode: SplatSkinningMode.DUAL_QUATERNION,
    });

    const identityQuat = new THREE.Quaternion();
    for (const bone of bones) {
      skinning.setRestQuatPos(bone.idx, identityQuat, new THREE.Vector3(...bone.pos));
    }

    for (let i = 0; i < Math.min(lbsWeights.length, skinning.numSplats); i++) {
      const w = lbsWeights[i];
      const pairs = w.map((val, idx) => [idx, val] as [number, number]).sort((a, b) => b[1] - a[1]);
      skinning.setSplatBones(
        i,
        new THREE.Vector4(pairs[0][0], pairs[1][0], pairs[2][0], pairs[3][0]),
        new THREE.Vector4(pairs[0][1], pairs[1][1], pairs[2][1], pairs[3][1]),
      );
    }

    (splatMesh as unknown as { skinning: unknown }).skinning = skinning;
    skinning.updateBones();
  }

  return { renderer, scene, camera, splatMesh, skinning, bones, canvas };
}
