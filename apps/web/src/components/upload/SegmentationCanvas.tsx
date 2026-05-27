'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { createMaskOverlay, segmentWithSAM } from '@/lib/sam-client';
import type { ClientSegmentationResult, SegmentationPoint } from '@/lib/sam-client';
import styles from './SegmentationCanvas.module.css';

interface SegmentationCanvasProps {
  imageUrl: string;
  onSegmented: (result: ClientSegmentationResult, imageData: ImageData) => void;
}

export function SegmentationCanvas({ imageUrl, onSegmented }: SegmentationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [points, setPoints] = useState<SegmentationPoint[]>([]);
  const [imgSize, setImgSize] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.width;
      canvas.height = img.height;
      setImgSize({ width: img.width, height: img.height });
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      setImageData(ctx.getImageData(0, 0, img.width, img.height));
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!imageData || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = imgSize.width / rect.width;
      const scaleY = imgSize.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const newPoint: SegmentationPoint = {
        x: Math.round(x),
        y: Math.round(y),
        label: 1,
      };
      const newPoints = [...points, newPoint];
      setPoints(newPoints);

      const result = await segmentWithSAM(imageData, newPoints);

      const ctx = canvasRef.current.getContext('2d')!;
      const overlay = createMaskOverlay(imageData, result.mask);
      ctx.putImageData(overlay, 0, 0);

      for (const p of newPoints) {
        ctx.fillStyle = p.label === 1 ? '#22c55e' : '#ef4444';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      onSegmented(result, imageData);
    },
    [imageData, points, imgSize, onSegmented]
  );

  return (
    <div className={styles.container}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onClick={handleClick}
      />
      <p className={styles.hint}>
        {points.length === 0
          ? 'Click on the head to segment it'
          : `${points.length} point(s) selected - click to refine`}
      </p>
    </div>
  );
}
