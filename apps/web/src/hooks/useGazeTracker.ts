'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { lerpEyePose, mouseToNDC, screenToGaze } from '@/lib/gaze-math';
import type { EyePose } from '@/types/gaze';

const NEUTRAL: EyePose = {
  left: { pitch: 0, yaw: 0 },
  right: { pitch: 0, yaw: 0 },
};

export function useGazeTracker(containerRef: React.RefObject<HTMLElement | null>) {
  const [eyePose, setEyePose] = useState<EyePose>(NEUTRAL);
  const targetRef = useRef<EyePose>(NEUTRAL);
  const currentRef = useRef<EyePose>(NEUTRAL);
  const rafRef = useRef<number>(0);

  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const ndc = mouseToNDC(e.clientX, e.clientY, rect);
      targetRef.current = screenToGaze(ndc);
    },
    [containerRef]
  );

  const handlePointerLeave = useCallback(() => {
    targetRef.current = NEUTRAL;
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    el.addEventListener('pointermove', handlePointerMove);
    el.addEventListener('pointerleave', handlePointerLeave);

    const animate = () => {
      currentRef.current = lerpEyePose(currentRef.current, targetRef.current);
      setEyePose({ ...currentRef.current });
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      el.removeEventListener('pointermove', handlePointerMove);
      el.removeEventListener('pointerleave', handlePointerLeave);
      cancelAnimationFrame(rafRef.current);
    };
  }, [containerRef, handlePointerMove, handlePointerLeave]);

  return eyePose;
}
