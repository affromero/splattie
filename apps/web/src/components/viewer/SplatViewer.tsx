'use client';

import { useRef } from 'react';
import { useGazeTracker } from '@/hooks/useGazeTracker';
import type { EyePose } from '@/types/gaze';
import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  spzUrl: string;
  onGazeUpdate?: (eyePose: EyePose) => void;
}

export function SplatViewer({ spzUrl, onGazeUpdate }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const eyePose = useGazeTracker(containerRef);

  if (onGazeUpdate) {
    onGazeUpdate(eyePose);
  }

  return (
    <div ref={containerRef} className={styles.container}>
      <canvas className={styles.canvas} />
      <div className={styles.info}>
        <span className={styles.badge}>3DGS</span>
        <span className={styles.gazeInfo}>
          L: {eyePose.left.yaw.toFixed(2)}, {eyePose.left.pitch.toFixed(2)} | R:{' '}
          {eyePose.right.yaw.toFixed(2)}, {eyePose.right.pitch.toFixed(2)}
        </span>
        <span className={styles.url}>{spzUrl.split('/').pop()}</span>
      </div>
    </div>
  );
}
