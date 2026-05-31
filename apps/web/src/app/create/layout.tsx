import type { Metadata } from 'next';
import type { ReactNode } from 'react';

// The /create page is a client component, so its metadata lives in this
// server-component layout segment.
export const metadata: Metadata = {
  title: 'Create your asset · Splattie',
  description:
    'Upload an image and generate an interactive rigged 3D asset that reacts to your visitors. One line of HTML to embed.',
  alternates: { canonical: '/create' },
  openGraph: {
    title: 'Create your asset · Splattie',
    description: 'Turn a portrait, full body, or object image into an interactive 3D asset.',
    url: '/create',
  },
};

export default function CreateLayout({ children }: { children: ReactNode }) {
  return children;
}
