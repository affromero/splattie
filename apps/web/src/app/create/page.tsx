'use client';

import Image from 'next/image';
import { useState } from 'react';
import styles from './page.module.css';

type Step = 'upload' | 'segment' | 'generate' | 'view';

export default function CreatePage() {
  const [step, setStep] = useState<Step>('upload');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setStep('segment');
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setStep('segment');
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

      {step === 'upload' && (
        <div
          className={styles.dropzone}
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
        >
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
        <div className={styles.segmentPreview}>
          <Image src={imagePreview} alt="Uploaded" className={styles.preview} width={500} height={400} />
          <p className={styles.hint}>
            Head segmentation will run automatically. Click below to confirm.
          </p>
          <button
            className={styles.button}
            onClick={() => setStep('generate')}
          >
            Confirm &amp; Generate
          </button>
        </div>
      )}

      {step === 'generate' && (
        <div className={styles.generating}>
          <div className={styles.spinner} />
          <p>Generating your 3D head...</p>
          <p className={styles.hint}>This takes about 5-10 seconds.</p>
        </div>
      )}

      {imageFile && null}
    </main>
  );
}
