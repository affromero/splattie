'use client';

import { useParams } from 'next/navigation';
import { SplatViewer } from '@/components/viewer/SplatViewer';
import styles from './page.module.css';

export default function ViewPage() {
  const params = useParams<{ id: string }>();
  const modelId = params.id;

  return (
    <main className={styles.page}>
      <SplatViewer viewerUrl={`/demo/viewer.html?zip=${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/storage/${modelId}/${modelId}.splattie`} />
    </main>
  );
}
