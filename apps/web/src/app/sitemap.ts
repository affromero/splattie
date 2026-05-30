import type { MetadataRoute } from 'next';
import { SELF_HOST, SITE_URL } from '@/lib/site';

/**
 * Static sitemap. User-generated avatar pages (/view/[id]) are intentionally
 * excluded — they are not publicly enumerable and shouldn't be crawled.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  const routes: { path: string; priority: number; changeFrequency: 'weekly' | 'monthly' }[] = [
    { path: '/', priority: 1, changeFrequency: 'weekly' },
    { path: '/editor.html', priority: 0.5, changeFrequency: 'monthly' },
  ];

  if (SELF_HOST) {
    routes.push({ path: '/create', priority: 0.7, changeFrequency: 'monthly' });
  }

  return routes.map((route) => ({
    url: `${SITE_URL}${route.path}`,
    lastModified,
    changeFrequency: route.changeFrequency,
    priority: route.priority,
  }));
}
