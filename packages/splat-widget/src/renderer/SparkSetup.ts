import * as THREE from 'three';

export interface SparkInstance {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  sparkRenderer: unknown;
  splatMesh: unknown;
  canvas: HTMLCanvasElement;
}

export async function createSparkInstance(
  container: HTMLElement,
  splatUrl: string,
  backgroundColor: number,
  onFrame?: (time: number, deltaTime: number) => void,
  onLoad?: () => void,
): Promise<SparkInstance> {
  const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');

  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const width = container.clientWidth;
  const height = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ canvas });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(backgroundColor);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);

  const sparkRenderer = new SparkRenderer({ renderer });

  const splatMesh = new SplatMesh({
    url: splatUrl,
    onLoad: () => onLoad?.(),
    onFrame: onFrame
      ? ({ time, deltaTime }: { time: number; deltaTime: number }) => onFrame(time, deltaTime)
      : undefined,
  });

  scene.add(splatMesh as unknown as THREE.Object3D);

  return { renderer, scene, camera, sparkRenderer, splatMesh, canvas };
}
