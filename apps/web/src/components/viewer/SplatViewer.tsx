'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useGazeTracker } from '@/hooks/useGazeTracker';
import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  spzUrl?: string;
}

export function SplatViewer({ spzUrl = '/demo/andres.ply' }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [statusMsg, setStatusMsg] = useState('initializing...');

  const headPose = useGazeTracker(containerRef);

  const initViewer = useCallback(async () => {
    const container = containerRef.current;
    if (!container) return;

    try {
      setStatusMsg('loading viewer...');
      const GS = await import('@mkkellogg/gaussian-splats-3d');

      const viewer = new GS.Viewer({
        selfDrivenMode: true,
        useBuiltInControls: false,
        rootElement: container,
        sceneRevealMode: GS.SceneRevealMode.Instant,
        logLevel: GS.LogLevel.None,
        sharedMemoryForWorkers: false,
      });

      viewerRef.current = viewer;

      setStatusMsg('loading splat...');
      await viewer.addSplatScene(spzUrl, {
        showLoadingUI: false,
        progressiveLoad: true,
      });

      setStatus('ready');
      setStatusMsg('ready');
    } catch (err) {
      setStatus('error');
      setStatusMsg(err instanceof Error ? err.message : String(err));
    }
  }, [spzUrl]);

  useEffect(() => {
    initViewer();
    return () => {
      const viewer = viewerRef.current as { dispose?: () => void } | null;
      viewer?.dispose?.();
      viewerRef.current = null;
    };
  }, [initViewer]);

  useEffect(() => {
    if (status !== 'ready') return;
    const viewer = viewerRef.current as {
      splatMesh?: { rotation?: { y: number; x: number } };
    } | null;
    if (viewer?.splatMesh?.rotation) {
      viewer.splatMesh.rotation.y = headPose.neckYaw * 0.8;
      viewer.splatMesh.rotation.x = -headPose.neckPitch * 0.5;
    }
  }, [status, headPose]);

  return (
    <div ref={containerRef} className={styles.container}>
      {status !== 'ready' && (
        <div className={styles.overlay}>
          <span className={status === 'error' ? styles.errorText : styles.loadingText}>
            {statusMsg}
          </span>
        </div>
      )}
    </div>
  );
}
