'use client';

import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

export default function DemoPage() {
  return (
    <main className={styles.page}>
      <div className={styles.viewerContainer}>
        <SplatViewer spzUrl="/demo/head.spz" />
      </div>
      <aside className={styles.aside}>
        <p className={styles.hint}>
          Move your cursor over the viewport. The head rotates to follow your gaze.
        </p>
        <p className={styles.meta}>
          .spz format · Spark 2.0 · FLAME LBS · client-side rendering
        </p>
      </aside>
    </main>
  );
}
