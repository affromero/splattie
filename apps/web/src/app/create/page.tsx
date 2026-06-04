'use client';

import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { generateFromUpload } from '@/lib/api-client';
import { track } from '@/lib/track';
import type { AssetType } from '@/types/api';
import styles from './page.module.css';

type Step = 'upload' | 'preview' | 'generate';

const ASSET_OPTIONS: { value: AssetType; label: string; hint: string }[] = [
  { value: 'head', label: 'Head', hint: 'clear, front-facing portrait' },
  { value: 'body', label: 'Body', hint: 'full body, head to feet' },
  { value: 'object', label: 'Object', hint: 'single isolated object' },
  { value: 'quadruped_mammal', label: 'Animal', hint: 'four-legged mammal (cat, dog, horse, deer)' },
];

const GENERATING_COPY: Record<AssetType, string> = {
  head: 'Reconstructing head model...',
  body: 'Reconstructing body model...',
  object: 'Reconstructing and rigging object...',
  quadruped_mammal: 'Reconstructing and rigging quadruped...',
};

export default function CreatePage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>('upload');
  const [assetType, setAssetType] = useState<AssetType>('head');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function handleFileSelect(file: File) {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setStep('preview');
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

  async function handleGenerate() {
    if (!imageFile) return;

    setStep('generate');
    setError(null);

    try {
      const result = await generateFromUpload(imageFile, assetType);
      track('avatar_create', '/create', { modelId: result.modelId });
      router.push(`/view/${result.modelId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
      setStep('preview');
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Create</h1>
        <p className={styles.subtitle}>
          Upload one image and choose the asset category.
        </p>
      </header>

      <div className={styles.assetTabs} role="tablist" aria-label="Asset type">
        {ASSET_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={assetType === option.value ? styles.assetTabActive : styles.assetTab}
            onClick={() => setAssetType(option.value)}
            aria-pressed={assetType === option.value}
          >
            <span>{option.label}</span>
            <small>{option.hint}</small>
          </button>
        ))}
      </div>

      <div className={styles.steps}>
        <span className={step === 'upload' ? styles.stepActive : styles.step}>upload</span>
        <span className={step === 'preview' ? styles.stepActive : styles.step}>preview</span>
        <span className={step === 'generate' ? styles.stepActive : styles.step}>generate</span>
      </div>

      {error && <p className={styles.error}>{error}</p>}

      {step === 'upload' && (
        <>
          <div
            className={styles.dropzone}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            <p>Drop an image here, or</p>
            <label className={styles.fileLabel}>
              Choose file
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className={styles.fileInput}
              />
            </label>
          </div>
          <p className={styles.privacy}>
            Your images are deleted immediately after processing. Nothing is stored.
          </p>
        </>
      )}

      {step === 'preview' && imagePreview && (
        <div className={styles.segmentSection}>
          <Image
            src={imagePreview}
            alt="Preview"
            width={400}
            height={400}
            className={styles.preview}
          />
          <div className={styles.actions}>
            <button className={styles.button} onClick={handleGenerate}>
              Generate 3D {assetType}
            </button>
            <button
              className={styles.buttonSecondary}
              onClick={() => { setStep('upload'); setImageFile(null); setImagePreview(null); }}
            >
              Choose different image
            </button>
          </div>
        </div>
      )}

      {step === 'generate' && (
        <div className={styles.generating}>
          <div className={styles.spinner} />
          <p>{GENERATING_COPY[assetType]}</p>
          <p className={styles.hint}>This takes about 30 seconds on GPU.</p>
        </div>
      )}
    </main>
  );
}
