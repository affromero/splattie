'use client';

import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

export default function DemoPage() {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Demo Viewer</h1>
        <p className={styles.subtitle}>Move your cursor over the 3D head to see the eyes follow.</p>
      </header>
      <div className={styles.viewerContainer}>
        <SplatViewer spzUrl="/demo/head.spz" />
      </div>
    </main>
  );
}
