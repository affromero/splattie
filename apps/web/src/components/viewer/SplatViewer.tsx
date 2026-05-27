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
      setStatusMsg('waiting for layout...');
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
        logLevel: GS.LogLevel.Info,
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: true,
      });

      viewerRef.current = viewer;

      setStatusMsg('loading model...');
      await viewer.addSplatScene(spzUrl, {
        showLoadingUI: false,
      });

      setStatus('ready');
      setStatusMsg('');
    } catch (err) {
      setStatus('error');
      setStatusMsg(err instanceof Error ? err.message : String(err));
      console.error('SplatViewer error:', err);
    }
  }, [spzUrl]);

  useEffect(() => {
    initViewer();
    return () => {
      try {
        const viewer = viewerRef.current as { dispose?: () => void } | null;
        viewer?.dispose?.();
      } catch {
        // ignore dispose errors
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
