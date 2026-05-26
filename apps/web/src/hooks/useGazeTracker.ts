'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { lerpHeadPose, mouseToNDC, screenToHeadPose } from '@/lib/gaze-math';
import type { HeadPose } from '@/types/gaze';

const NEUTRAL: HeadPose = {
  eyes: {
    left: { pitch: 0, yaw: 0 },
    right: { pitch: 0, yaw: 0 },
  },
  neckYaw: 0,
  neckPitch: 0,
  jawOpen: 0,
};

export function useGazeTracker(containerRef: React.RefObject<HTMLElement | null>) {
  const [headPose, setHeadPose] = useState<HeadPose>(NEUTRAL);
  const targetRef = useRef<HeadPose>(NEUTRAL);
  const currentRef = useRef<HeadPose>(NEUTRAL);
  const rafRef = useRef<number>(0);

  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const ndc = mouseToNDC(e.clientX, e.clientY, rect);
      targetRef.current = screenToHeadPose(ndc);
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
      currentRef.current = lerpHeadPose(currentRef.current, targetRef.current);
      setHeadPose({ ...currentRef.current });
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      el.removeEventListener('pointermove', handlePointerMove);
      el.removeEventListener('pointerleave', handlePointerLeave);
      cancelAnimationFrame(rafRef.current);
    };
  }, [containerRef, handlePointerMove, handlePointerLeave]);

  return headPose;
}
