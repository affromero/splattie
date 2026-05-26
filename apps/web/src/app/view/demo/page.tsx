'use client';

import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

export default function DemoPage() {
  return (
    <main className={styles.page}>
      <div className={styles.viewerContainer}>
        <SplatViewer spzUrl="/demo/head.spz" />
      </div>
    </main>
  );
}
