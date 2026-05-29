import type { Metadata } from 'next';
import { JetBrains_Mono, Space_Grotesk } from 'next/font/google';
import { Analytics } from '@/components/analytics/Analytics';
import { Nav } from '@/components/layout/Nav';
import { SITE_URL } from '@/lib/site';
import '@/styles/globals.css';

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Splattie - Editor & widget for interactive 3D avatars',
  description: 'Open-source, in-browser editor and web component for interactive 3D Gaussian-splat avatars. Tune gaze, pose, and background, then embed with one line of HTML.',
  metadataBase: new URL(SITE_URL),
  alternates: { canonical: '/' },
  icons: {
    icon: '/favicon.svg',
    apple: '/logo.svg',
  },
  openGraph: {
    title: 'Splattie - Editor & widget for interactive 3D avatars',
    description: 'Open-source, in-browser editor and web component for interactive 3D Gaussian-splat avatars. Tune gaze, pose, and background, then embed with one line of HTML.',
    url: SITE_URL,
    siteName: 'Splattie',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Splattie - Editor & widget for interactive 3D avatars',
    description: 'In-browser editor and web component for interactive 3D Gaussian-splat avatars. Embed with one line of HTML.',
  },
  robots: { index: true, follow: true },
  // Set GOOGLE_SITE_VERIFICATION in the web container env to verify Search Console.
  verification: { google: process.env.GOOGLE_SITE_VERIFICATION },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body>
        <Analytics />
        <Nav />
        {children}
      </body>
    </html>
  );
}
