'use client';

import { usePathname } from 'next/navigation';
import { useEffect, useRef } from 'react';
import { track } from '@/lib/track';

/**
 * Records a pageview on first load and on every client-side route change.
 * Mounted once in the root layout. Renders nothing.
 */
export function Analytics() {
  const pathname = usePathname();
  const lastPath = useRef<string | null>(null);

  useEffect(() => {
    if (!pathname || lastPath.current === pathname) return;
    lastPath.current = pathname;
    track('pageview', pathname);
  }, [pathname]);

  return null;
}
