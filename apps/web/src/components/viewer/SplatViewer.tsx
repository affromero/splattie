'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { useGazeTracker } from '@/hooks/useGazeTracker';
import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  spzUrl?: string;
}

export function SplatViewer({ spzUrl = '/demo/head.spz' }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const splatMeshRef = useRef<THREE.Object3D | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [errorMsg, setErrorMsg] = useState('');

  const headPose = useGazeTracker(containerRef);

  const initSpark = useCallback(async () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;

    try {
      const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');

      const threeRenderer = new THREE.WebGLRenderer({ canvas });
      threeRenderer.setSize(width, height);
      threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      threeRenderer.setClearColor(0x0e0e14);
      rendererRef.current = threeRenderer;

      const scene = new THREE.Scene();
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
      camera.position.set(0, 0.1, 2.0);
      camera.lookAt(0, 0, 0);
      cameraRef.current = camera;

      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const _sparkRenderer = new SparkRenderer({
        renderer: threeRenderer,
      });

      const splatMesh = new SplatMesh({
        url: spzUrl,
        onLoad: () => {
          setStatus('ready');
        },
      });

      scene.add(splatMesh);
      splatMeshRef.current = splatMesh;

      setStatus('ready');
    } catch (err) {
      setStatus('error');
      setErrorMsg(err instanceof Error ? err.message : 'Failed to initialize Spark');
    }
  }, [spzUrl]);

  useEffect(() => {
    initSpark();

    const handleResize = () => {
      const container = containerRef.current;
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      rendererRef.current.setSize(w, h);
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      rendererRef.current?.dispose();
    };
  }, [initSpark]);

  useEffect(() => {
    if (status !== 'ready') return;

    const animate = () => {
      const mesh = splatMeshRef.current;
      const renderer = rendererRef.current;
      const scene = sceneRef.current;
      const camera = cameraRef.current;

      if (mesh && renderer && scene && camera) {
        mesh.rotation.y = headPose.neckYaw * 0.8;
        mesh.rotation.x = -headPose.neckPitch * 0.5;

        renderer.render(scene, camera);
      }

      rafId = requestAnimationFrame(animate);
    };

    let rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [status, headPose]);

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      {status === 'loading' && (
        <div className={styles.overlay}>
          <span className={styles.loadingText}>loading splat...</span>
        </div>
      )}
      {status === 'error' && (
        <div className={styles.overlay}>
          <span className={styles.errorText}>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}
