import type { Metadata } from 'next';
import type { ReactNode } from 'react';

// The /create page is a client component, so its metadata lives in this
// server-component layout segment.
export const metadata: Metadata = {
  title: 'Create your avatar · Splattie',
  description:
    'Upload a photo and generate an interactive 3D avatar whose eyes follow your visitors. One line of HTML to embed.',
  alternates: { canonical: '/create' },
  openGraph: {
    title: 'Create your avatar · Splattie',
    description: 'Turn a portrait into a living 3D avatar. Eyes follow visitors.',
    url: '/create',
  },
};

export default function CreateLayout({ children }: { children: ReactNode }) {
  return children;
}
