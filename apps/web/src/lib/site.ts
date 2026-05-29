/** Canonical site origin, used by metadata, sitemap, and robots. */
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://splattie.app';

/** Whether the self-host build (with the /create pipeline) is enabled. */
export const SELF_HOST = process.env.NEXT_PUBLIC_SELF_HOST === 'true';
