'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { useGazeTracker } from '@/hooks/useGazeTracker';
import { applyHeadPose, generateProceduralHead } from '@/lib/flame-animator';
import type { FlameParams } from '@/types/gaze';
import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  spzUrl?: string;
  flameParamsUrl?: string;
  proceduralCount?: number;
}

export function SplatViewer({ proceduralCount = 30_000 }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const headDataRef = useRef<{
    canonical: Float32Array;
    lbsWeights: Float32Array;
    flameParams: FlameParams;
  } | null>(null);
  const [ready, setReady] = useState(false);

  const headPose = useGazeTracker(containerRef);

  const initScene = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x12121a);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 10);
    camera.position.set(0, 0, 0.35);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const { positions, colors, lbsWeights, flameParams } =
      generateProceduralHead(proceduralCount);
    headDataRef.current = {
      canonical: new Float32Array(positions),
      lbsWeights,
      flameParams,
    };

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const colorArray = new Float32Array(proceduralCount * 3);
    for (let i = 0; i < proceduralCount; i++) {
      colorArray[i * 3] = colors[i * 4] / 255;
      colorArray[i * 3 + 1] = colors[i * 4 + 1] / 255;
      colorArray[i * 3 + 2] = colors[i * 4 + 2] / 255;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

    const material = new THREE.PointsMaterial({
      size: 0.003,
      vertexColors: true,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);
    pointsRef.current = points;

    setReady(true);
  }, [proceduralCount]);

  useEffect(() => {
    initScene();

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
  }, [initScene]);

  useEffect(() => {
    if (!ready) return;

    const animate = () => {
      const points = pointsRef.current;
      const headData = headDataRef.current;
      const renderer = rendererRef.current;
      const scene = sceneRef.current;
      const camera = cameraRef.current;

      if (points && headData && renderer && scene && camera) {
        const deformed = applyHeadPose(
          headData.canonical,
          headData.lbsWeights,
          headData.flameParams,
          headPose
        );

        const posAttr = points.geometry.getAttribute('position') as THREE.BufferAttribute;
        posAttr.array.set(deformed);
        posAttr.needsUpdate = true;

        renderer.render(scene, camera);
      }

      rafId = requestAnimationFrame(animate);
    };

    let rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [ready, headPose]);

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      <div className={styles.info}>
        <span className={styles.badge}>3DGS</span>
        <span className={styles.gazeInfo}>
          gaze: {headPose.eyes.left.yaw.toFixed(2)},{headPose.eyes.left.pitch.toFixed(2)} | neck:{' '}
          {headPose.neckYaw.toFixed(2)} | jaw: {headPose.jawOpen.toFixed(2)}
        </span>
        <span className={styles.url}>{proceduralCount.toLocaleString()} gaussians</span>
      </div>
    </div>
  );
}
