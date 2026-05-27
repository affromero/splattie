'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  spzUrl?: string;
}

export function SplatViewer({ spzUrl = '/demo/andres.ply' }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [statusMsg, setStatusMsg] = useState('initializing...');

  const initViewer = useCallback(async () => {
    const container = containerRef.current;
    if (!container) return;
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      setTimeout(() => initViewer(), 100);
      return;
    }

    try {
      setStatusMsg('loading viewer...');
      const GS = await import('@mkkellogg/gaussian-splats-3d');

      setStatusMsg('creating viewer...');
      const viewer = new GS.Viewer({
        selfDrivenMode: true,
        useBuiltInControls: true,
        rootElement: container,
        sceneRevealMode: GS.SceneRevealMode.Instant,
        logLevel: GS.LogLevel.Debug,
        sharedMemoryForWorkers: false,
        initialCameraPosition: [0, 0.02, 0.5],
        initialCameraLookAt: [0, 0, 0],
      });

      viewerRef.current = viewer;

      setStatusMsg('loading model...');
      await viewer.addSplatScene(spzUrl, {
        showLoadingUI: false,
        splatAlphaRemovalThreshold: 1,
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
      });

      setStatus('ready');
      setStatusMsg('');
    } catch (err) {
      setStatus('error');
      setStatusMsg(err instanceof Error ? err.message : String(err));
      console.error('SplatViewer init error:', err);
    }
  }, [spzUrl]);

  useEffect(() => {
    initViewer();
    return () => {
      try {
        const viewer = viewerRef.current as { dispose?: () => void } | null;
        viewer?.dispose?.();
      } catch {
        // ignore
      }
      viewerRef.current = null;
    };
  }, [initViewer]);

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
