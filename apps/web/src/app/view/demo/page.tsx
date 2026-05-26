'use client';

import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

export default function DemoPage() {
  return (
    <main className={styles.page}>
      <div className={styles.viewerContainer}>
        <SplatViewer proceduralCount={40000} />
      </div>
      <aside className={styles.aside}>
        <p className={styles.hint}>
          Move your cursor over the viewport. The head follows your gaze and reacts
          when the cursor enters the face region.
        </p>
        <p className={styles.meta}>
          40k gaussians · FLAME LBS · 60fps client-side
        </p>
      </aside>
    </main>
  );
}
