'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './page.module.css';

const SELF_HOST = process.env.NEXT_PUBLIC_SELF_HOST === 'true';

interface DemoFace {
  id: string;
  thumb: string;
  splattie: string;
  photographer: string;
  pexelsUrl: string;
}

const DEMO_FACES: DemoFace[] = [
  { id: '3762763', thumb: '/demos/thumbs/3762763.jpg', splattie: '/demos/3762763.splattie', photographer: 'Shiny Diamond', pexelsUrl: 'https://www.pexels.com/photo/3762763/' },
  { id: '3754430', thumb: '/demos/thumbs/3754430.jpg', splattie: '/demos/3754430.splattie', photographer: 'TUBARONES PHOTOGRAPHY', pexelsUrl: 'https://www.pexels.com/photo/3754430/' },
  { id: '7705909', thumb: '/demos/thumbs/7705909.jpg', splattie: '/demos/7705909.splattie', photographer: 'ShotPot', pexelsUrl: 'https://www.pexels.com/photo/7705909/' },
  { id: '8727488', thumb: '/demos/thumbs/8727488.jpg', splattie: '/demos/8727488.splattie', photographer: 'Tima Miroshnichenko', pexelsUrl: 'https://www.pexels.com/photo/8727488/' },
  { id: '8727554', thumb: '/demos/thumbs/8727554.jpg', splattie: '/demos/8727554.splattie', photographer: 'Tima Miroshnichenko', pexelsUrl: 'https://www.pexels.com/photo/8727554/' },
  { id: '35466969', thumb: '/demos/thumbs/35466969.jpg', splattie: '/demos/35466969.splattie', photographer: 'Daniel Hoffman Jackson', pexelsUrl: 'https://www.pexels.com/photo/35466969/' },
];

export default function Home() {
  const [activeFace, setActiveFace] = useState<DemoFace | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  const handleCardClick = useCallback((face: DemoFace) => {
    setActiveFace((prev) => (prev?.id === face.id ? null : face));
  }, []);

  useEffect(() => {
    if (activeFace && editorRef.current) {
      editorRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [activeFace]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setActiveFace(null);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <h1 className={styles.title}>
          Edit the splat.<br />
          <span className={styles.titleAccent}>Embed the avatar.</span>
        </h1>
        <p className={styles.subtitle}>
          An open-source, in-browser editor and web component for interactive
          3D Gaussian-splat avatars. Tune the gaze, pose, and background, then
          embed with one line of HTML. It reacts to your visitors.
        </p>
        <div className={styles.heroBadges}>
          <a href="https://github.com/affromero/splattie" target="_blank" rel="noopener noreferrer" className={styles.badge}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.4 3-.405 1.02.005 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" /></svg>
            Open source · MIT
          </a>
          <a href="https://www.npmjs.com/package/@afromero/splattie-widget" target="_blank" rel="noopener noreferrer" className={styles.badge}>
            @afromero/splattie-widget
          </a>
        </div>
      </section>

      <section className={styles.gallerySection}>
        <h2 className={styles.sectionTitle}>Try the editor</h2>
        <p className={styles.sectionSubtitle}>Click any portrait to open it in the editor.</p>
        <a href="/editor.html" className={styles.editorLink}>
          Or drop in your own .splattie ↗
        </a>

        <div className={styles.gallery}>
          {DEMO_FACES.map((face) => (
            <div key={face.id} className={styles.cardWrapper}>
              <button
                className={`${styles.card} ${activeFace?.id === face.id ? styles.cardActive : ''}`}
                onClick={() => handleCardClick(face)}
              >
                <Image
                  src={face.thumb}
                  alt={`Portrait by ${face.photographer}`}
                  width={400}
                  height={500}
                  className={styles.cardImage}
                />
              </button>
              <a
                href={face.pexelsUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.credit}
              >
                {face.photographer}
              </a>
            </div>
          ))}
        </div>

        {activeFace && (
          <div className={styles.editorSection} ref={editorRef}>
            <div className={styles.editorHeader}>
              <span className={styles.editorLabel}>
                Editor - {activeFace.photographer}
              </span>
              <button className={styles.editorClose} onClick={() => setActiveFace(null)}>
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                  <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <iframe
              key={activeFace.id}
              src={`/editor.html?src=${activeFace.splattie}`}
              className={styles.editorFrame}
              title={`Editor for ${activeFace.photographer}'s portrait`}
            />
          </div>
        )}
      </section>

      <section className={styles.howSection}>
        <h2 className={styles.sectionTitle}>How it works</h2>
        <div className={styles.steps}>
          <div className={styles.step}>
            <div className={styles.stepNumber}>1</div>
            <h3 className={styles.stepTitle}>Open</h3>
            <p className={styles.stepDesc}>Start from a demo, or drop your own .splattie into the editor</p>
          </div>
          <div className={styles.stepDivider} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <h3 className={styles.stepTitle}>Edit</h3>
            <p className={styles.stepDesc}>Tune gaze, pose, idle motion, and background — all in the browser</p>
          </div>
          <div className={styles.stepDivider} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>3</div>
            <h3 className={styles.stepTitle}>Embed</h3>
            <p className={styles.stepDesc}>One line of HTML. It reacts to your visitors.</p>
          </div>
        </div>
      </section>

      <section className={styles.embedSection}>
        <h2 className={styles.sectionTitle}>The widget</h2>
        <p className={styles.sectionSubtitle}>
          A standalone, MIT-licensed web component — its own repo, drop it into any site or framework.
        </p>
        <pre className={styles.codeBlock}><code>{`<splattie-widget src="avatar.splattie" />`}</code></pre>
        <div className={styles.heroBadges}>
          <a
            href="https://github.com/affromero/splattie-widget"
            target="_blank"
            rel="noopener noreferrer"
            className={styles.badge}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.4 3-.405 1.02.005 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" /></svg>
            github.com/affromero/splattie-widget
          </a>
          <a
            href="https://www.npmjs.com/package/@afromero/splattie-widget"
            target="_blank"
            rel="noopener noreferrer"
            className={styles.badge}
          >
            @afromero/splattie-widget
          </a>
        </div>
      </section>

      {SELF_HOST && (
        <section className={styles.uploadSection}>
          <Link href="/create" className={styles.uploadBoxActive}>
            <h2 className={styles.uploadTitle}>Upload your own photo</h2>
            <p className={styles.uploadSubtitleActive}>Generate a 3D head on your local GPU</p>
            <div className={styles.uploadPlaceholderActive}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <path d="M24 8v32M8 24h32" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
          </Link>
        </section>
      )}

      <footer className={styles.footer}>
        <p>
          <a href="https://github.com/affromero/splattie" target="_blank" rel="noopener noreferrer">GitHub</a>{' · '}
          <a href="https://www.npmjs.com/package/@afromero/splattie-widget" target="_blank" rel="noopener noreferrer">npm</a>{' · '}
          <a href="https://github.com/affromero/splattie-widget/blob/main/FORMAT.md" target="_blank" rel="noopener noreferrer">format spec</a>{' · '}
          <a href="https://github.com/affromero/splattie/issues" target="_blank" rel="noopener noreferrer">feedback</a>
        </p>
        <p className={styles.footerMuted}>
          Built with <a href="https://github.com/aigc3d/LAM" target="_blank" rel="noopener noreferrer">LAM</a>{' '}
          + <a href="https://sparkjs.dev" target="_blank" rel="noopener noreferrer">Spark</a>{' '}
          + <a href="https://flame.is.tue.mpg.de" target="_blank" rel="noopener noreferrer">FLAME</a>{'. '}
          Photos from <a href="https://www.pexels.com" target="_blank" rel="noopener noreferrer">Pexels</a>.
        </p>
      </footer>
    </main>
  );
}
