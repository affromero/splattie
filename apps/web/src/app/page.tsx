'use client';

import Image from 'next/image';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './page.module.css';

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
          One photo in.<br />
          <span className={styles.titleAccent}>Interactive 3D avatar out.</span>
        </h1>
        <p className={styles.subtitle}>
          Turn any portrait into a living avatar for your website.
          Eyes follow visitors. Head reacts to hover. One line of HTML.
        </p>
      </section>

      <section className={styles.gallerySection}>
        <h2 className={styles.sectionTitle}>Try it</h2>
        <p className={styles.sectionSubtitle}>Click a portrait to bring it to life</p>

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
                Editor &mdash; {activeFace.photographer}
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
            <h3 className={styles.stepTitle}>Upload</h3>
            <p className={styles.stepDesc}>Any front-facing portrait photo</p>
          </div>
          <div className={styles.stepDivider} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <h3 className={styles.stepTitle}>Generate</h3>
            <p className={styles.stepDesc}>LAM reconstructs a 3D Gaussian head in ~30s</p>
          </div>
          <div className={styles.stepDivider} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>3</div>
            <h3 className={styles.stepTitle}>Embed</h3>
            <p className={styles.stepDesc}>One line of HTML. Eyes follow visitors.</p>
          </div>
        </div>
      </section>

      <section className={styles.embedSection}>
        <h2 className={styles.sectionTitle}>Add to your site</h2>
        <pre className={styles.codeBlock}><code>{`<splattie-widget src="avatar.splattie" />`}</code></pre>
        <p className={styles.embedNote}>
          MIT licensed web component. Works in any framework.
        </p>
      </section>

      <section className={styles.uploadSection}>
        <div className={styles.uploadBox}>
          <h2 className={styles.uploadTitle}>Upload your own photo</h2>
          <p className={styles.uploadSubtitle}>Coming soon</p>
          <div className={styles.uploadPlaceholder}>
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <path d="M24 8v32M8 24h32" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.3" />
            </svg>
          </div>
        </div>
      </section>

      <footer className={styles.footer}>
        <p>
          Built with <a href="https://github.com/aigc3d/LAM" target="_blank" rel="noopener noreferrer">LAM</a>{' '}
          + <a href="https://sparkjs.dev" target="_blank" rel="noopener noreferrer">Spark</a>{' '}
          + <a href="https://flame.is.tue.mpg.de" target="_blank" rel="noopener noreferrer">FLAME</a>
        </p>
        <p className={styles.footerMuted}>
          Photos from <a href="https://www.pexels.com" target="_blank" rel="noopener noreferrer">Pexels</a> (free license)
        </p>
      </footer>
    </main>
  );
}
