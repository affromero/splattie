import type { Metadata } from 'next';
import type { ReactNode } from 'react';

// The /view/[id] page is a client component; per-avatar metadata lives here.
export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  return {
    title: 'Interactive 3D avatar · Splattie',
    description: 'An interactive 3D avatar whose eyes follow your cursor, made with Splattie.',
    alternates: { canonical: `/view/${id}` },
    openGraph: {
      title: 'Interactive 3D avatar · Splattie',
      description: 'Eyes follow your cursor. Made with Splattie.',
      url: `/view/${id}`,
    },
  };
}

export default function ViewLayout({ children }: { children: ReactNode }) {
  return children;
}
