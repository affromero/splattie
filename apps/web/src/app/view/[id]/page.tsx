'use client';

import { useParams } from 'next/navigation';
import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function ViewPage() {
  const params = useParams<{ id: string }>();
  const modelId = params.id;

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Your 3D Head</h1>
        <p className={styles.subtitle}>Move your cursor to control eye gaze.</p>
      </header>
      <div className={styles.viewerContainer}>
        <SplatViewer
          spzUrl={`${API_URL}/storage/${modelId}/head.spz`}
          flameParamsUrl={`${API_URL}/storage/${modelId}/flame.json`}
        />
      </div>
    </main>
  );
}
