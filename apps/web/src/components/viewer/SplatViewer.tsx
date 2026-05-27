'use client';

import styles from './SplatViewer.module.css';

interface SplatViewerProps {
  viewerUrl?: string;
}

export function SplatViewer({ viewerUrl = '/demo/viewer.html' }: SplatViewerProps) {
  return (
    <iframe
      src={viewerUrl}
      className={styles.viewer}
      title="3D Head Viewer"
    />
  );
}
