'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useState } from 'react';
import { SegmentationCanvas } from '@/components/upload/SegmentationCanvas';
import { generateHead, segmentImage } from '@/lib/api-client';
import type { ClientSegmentationResult } from '@/lib/sam-client';
import type { GenerationProgress } from '@/types/api';
import styles from './page.module.css';

type Step = 'upload' | 'segment' | 'generate' | 'view';

export default function CreatePage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>('upload');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [segResult, setSegResult] = useState<ClientSegmentationResult | null>(null);
  const [originalImageData, setOriginalImageData] = useState<ImageData | null>(null);
  const [progress, setProgress] = useState<GenerationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileSelect(file: File) {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setStep('segment');
    setError(null);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }

  const handleSegmented = useCallback(
    (result: ClientSegmentationResult, imgData: ImageData) => {
      setSegResult(result);
      setOriginalImageData(imgData);
    },
    []
  );

  async function handleGenerate() {
    if (!imageFile || !segResult || !originalImageData) return;

    setStep('generate');
    setError(null);

    try {
      const segResponse = await segmentImage(imageFile);

      const result = await generateHead(
        segResponse.previewUrl.replace('/preview.png', '/original.png'),
        segResponse.maskUrl,
        (p) => setProgress(p)
      );

      router.push(`/view/${result.modelId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
      setStep('segment');
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Create your 3D head</h1>
        <p className={styles.subtitle}>Upload a photo with a clear face to get started.</p>
      </header>

      <div className={styles.steps}>
        <span className={step === 'upload' ? styles.stepActive : styles.step}>Upload</span>
        <span className={step === 'segment' ? styles.stepActive : styles.step}>Segment</span>
        <span className={step === 'generate' ? styles.stepActive : styles.step}>Generate</span>
        <span className={step === 'view' ? styles.stepActive : styles.step}>View</span>
      </div>

      {error && <p className={styles.error}>{error}</p>}

      {step === 'upload' && (
        <div className={styles.dropzone} onDrop={handleDrop} onDragOver={(e) => e.preventDefault()}>
          <p>Drag and drop a photo here, or</p>
          <label className={styles.fileLabel}>
            Browse files
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className={styles.fileInput}
            />
          </label>
        </div>
      )}

      {step === 'segment' && imagePreview && (
        <div className={styles.segmentSection}>
          <SegmentationCanvas imageUrl={imagePreview} onSegmented={handleSegmented} />
          {segResult && (
            <button className={styles.button} onClick={handleGenerate}>
              Generate 3D Head
            </button>
          )}
        </div>
      )}

      {step === 'generate' && (
        <div className={styles.generating}>
          <div className={styles.spinner} />
          <p>Generating your 3D head...</p>
          {progress && (
            <div className={styles.progressContainer}>
              <div className={styles.progressBar} style={{ width: `${progress.pct}%` }} />
              <span className={styles.progressText}>{progress.stage} ({progress.pct}%)</span>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
