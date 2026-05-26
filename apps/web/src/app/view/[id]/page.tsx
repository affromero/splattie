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
      <div className={styles.viewerContainer}>
        <SplatViewer spzUrl={`${API_URL}/storage/${modelId}/head.spz`} />
      </div>
    </main>
  );
}
