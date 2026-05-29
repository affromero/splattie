'use client';

import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import styles from './page.module.css';

const SELF_HOST = process.env.NEXT_PUBLIC_SELF_HOST === 'true';

type Category = 'head' | 'body';

interface Demo {
  id: string;
  category: Category;
  photographer: string;
  pexelsUrl: string;
}

// Demos live in category subfolders: /demos/{heads,bodies}/<id>.{jpg,splattie}.
const folder = (c: Category): string => (c === 'head' ? 'heads' : 'bodies');
const demoThumb = (d: Demo): string => `/demos/${folder(d.category)}/${d.id}.jpg`;
const demoSrc = (d: Demo): string => `/demos/${folder(d.category)}/${d.id}.splattie`;

const DEMOS: Demo[] = [
  // Heads (LAM)
  { id: '3762763', category: 'head', photographer: 'Shiny Diamond', pexelsUrl: 'https://www.pexels.com/photo/3762763/' },
  { id: '3754430', category: 'head', photographer: 'TUBARONES PHOTOGRAPHY', pexelsUrl: 'https://www.pexels.com/photo/3754430/' },
  { id: '7705909', category: 'head', photographer: 'ShotPot', pexelsUrl: 'https://www.pexels.com/photo/7705909/' },
  { id: '8727488', category: 'head', photographer: 'Tima Miroshnichenko', pexelsUrl: 'https://www.pexels.com/photo/8727488/' },
  { id: '8727554', category: 'head', photographer: 'Tima Miroshnichenko', pexelsUrl: 'https://www.pexels.com/photo/8727554/' },
  { id: '35466969', category: 'head', photographer: 'Daniel Hoffman Jackson', pexelsUrl: 'https://www.pexels.com/photo/35466969/' },
  // Bodies (LHM)
  { id: '6668809', category: 'body', photographer: 'N. Voitkevich', pexelsUrl: 'https://www.pexels.com/photo/6668809/' },
  { id: '31894858', category: 'body', photographer: 'Andrea Musto', pexelsUrl: 'https://www.pexels.com/photo/31894858/' },
  { id: '31969138', category: 'body', photographer: 'Sephina Cornwall', pexelsUrl: 'https://www.pexels.com/photo/31969138/' },
  { id: '8217520', category: 'body', photographer: 'Mart Production', pexelsUrl: 'https://www.pexels.com/photo/8217520/' },
  { id: '6211660', category: 'body', photographer: 'Dima Valkov', pexelsUrl: 'https://www.pexels.com/photo/6211660/' },
  { id: '18516993', category: 'body', photographer: 'Foto Art Events', pexelsUrl: 'https://www.pexels.com/photo/18516993/' },
];

const HEADS = DEMOS.filter((d) => d.category === 'head');
const BODIES = DEMOS.filter((d) => d.category === 'body');

function Carousel({
  category,
  demos,
  activeId,
  paused,
  onSelect,
}: {
  category: Category;
  demos: Demo[];
  activeId: string | null;
  paused: boolean;
  onSelect: (demo: Demo) => void;
}) {
  // Duplicate the list so auto-scroll loops seamlessly (reset at half the width).
  const loop = [...demos, ...demos];
  const scrollRef = useRef<HTMLDivElement>(null);
  const hoverRef = useRef(false);

  // Native horizontal scroll (trackpad/drag works any time) + JS auto-scroll on top.
  // Auto-scroll pauses while hovered or while an avatar is selected, so you can
  // browse and click another one.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return undefined;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined;
    let raf = 0;
    const tick = () => {
      if (!paused && !hoverRef.current) {
        const half = el.scrollWidth / 2;
        el.scrollLeft = half > 0 && el.scrollLeft >= half ? el.scrollLeft - half + 0.4 : el.scrollLeft + 0.4;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [paused]);

  return (
    <div
      ref={scrollRef}
      className={styles.carousel}
      data-category={category}
      onPointerEnter={() => { hoverRef.current = true; }}
      onPointerLeave={() => { hoverRef.current = false; }}
    >
      <div className={styles.carouselTrack}>
        {loop.map((demo, i) => (
          <div key={`${demo.id}-${i}`} className={styles.carouselItem}>
            <button
              className={`${styles.carouselCard} ${activeId === demo.id ? styles.cardActive : ''}`}
              onClick={() => onSelect(demo)}
              aria-label={`Bring ${demo.category} by ${demo.photographer} to life`}
            >
              {/* Plain img (not next/image): the loop duplicates nodes + lazy-loads. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={demoThumb(demo)} alt={`${demo.category} by ${demo.photographer}`} className={styles.carouselImg} loading="lazy" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [activeDemo, setActiveDemo] = useState<Demo | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  const handleSelect = useCallback((demo: Demo) => {
    setActiveDemo((prev) => (prev?.id === demo.id ? null : demo));
  }, []);

  useEffect(() => {
    if (activeDemo && editorRef.current) {
      editorRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeDemo]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setActiveDemo(null);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Selecting any avatar pauses both carousels so the chosen one stays in view.
  const paused = activeDemo !== null;

  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <h1 className={styles.title}>
          One photo in.<br />
          <span className={styles.titleAccent}>Interactive 3D avatar out.</span>
        </h1>
        <p className={styles.subtitle}>
          Turn any portrait — or full body — into a living avatar for your website.
          It looks at your visitors and reacts to hover. One line of HTML.
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
        <h2 className={styles.sectionTitle}>Try it</h2>
        <p className={styles.sectionSubtitle}>Click an avatar to bring it to life — the carousel pauses while you play</p>

        <div className={styles.categoryLabel}>Heads<span className={styles.categoryHint}>eyes follow your cursor</span></div>
        <Carousel category="head" demos={HEADS} activeId={activeDemo?.id ?? null} paused={paused} onSelect={handleSelect} />

        <div className={styles.categoryLabel}>Bodies<span className={styles.categoryHint}>head &amp; torso turn toward you</span></div>
        <Carousel category="body" demos={BODIES} activeId={activeDemo?.id ?? null} paused={paused} onSelect={handleSelect} />

        {activeDemo && (
          <div className={styles.editorSection} ref={editorRef}>
            <div className={styles.editorHeader}>
              <span className={styles.editorLabel}>
                Editor · {activeDemo.category} ·{' '}
                <a href={activeDemo.pexelsUrl} target="_blank" rel="noopener noreferrer">{activeDemo.photographer}</a>
              </span>
              <button className={styles.editorClose} onClick={() => setActiveDemo(null)} aria-label="Close editor">
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                  <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <iframe
              key={activeDemo.id}
              src={`/editor.html?src=${demoSrc(activeDemo)}`}
              className={styles.editorFrame}
              title={`Editor for ${activeDemo.photographer}'s ${activeDemo.category}`}
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
            <p className={styles.stepDesc}>A portrait or a full-body photo</p>
          </div>
          <div className={styles.stepDivider} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <h3 className={styles.stepTitle}>Generate</h3>
            <p className={styles.stepDesc}>LAM (heads) or LHM (bodies) reconstructs a 3D Gaussian avatar</p>
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
        <h2 className={styles.sectionTitle}>Add to your site</h2>
        <pre className={styles.codeBlock}><code>{`<splattie-widget src="avatar.splattie" />`}</code></pre>
        <p className={styles.embedNote}>
          MIT licensed web component. Works in any framework.
        </p>
      </section>

      <section className={styles.uploadSection}>
        {SELF_HOST ? (
          <Link href="/create" className={styles.uploadBoxActive}>
            <h2 className={styles.uploadTitle}>Upload your own photo</h2>
            <p className={styles.uploadSubtitleActive}>Generate a 3D avatar on your local GPU</p>
            <div className={styles.uploadPlaceholderActive}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <path d="M24 8v32M8 24h32" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
          </Link>
        ) : (
          <div className={styles.uploadBox}>
            <h2 className={styles.uploadTitle}>Upload your own photo</h2>
            <p className={styles.uploadSubtitle}>Coming soon</p>
            <div className={styles.uploadPlaceholder}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <path d="M24 8v32M8 24h32" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.3" />
              </svg>
            </div>
          </div>
        )}
      </section>

      <footer className={styles.footer}>
        <p>
          <a href="https://github.com/affromero/splattie" target="_blank" rel="noopener noreferrer">GitHub</a>{' · '}
          <a href="https://www.npmjs.com/package/@afromero/splattie-widget" target="_blank" rel="noopener noreferrer">npm</a>{' · '}
          <a href="https://github.com/affromero/splattie-widget/blob/main/FORMAT.md" target="_blank" rel="noopener noreferrer">format spec</a>{' · '}
          <a href="https://github.com/affromero/splattie/issues" target="_blank" rel="noopener noreferrer">feedback</a>
        </p>
        <p className={styles.footerMuted}>
          Built with <a href="https://github.com/aigc3d/LAM" target="_blank" rel="noopener noreferrer">LAM</a>{' '}
          + <a href="https://github.com/aigc3d/LHM" target="_blank" rel="noopener noreferrer">LHM</a>{' '}
          + <a href="https://sparkjs.dev" target="_blank" rel="noopener noreferrer">Spark</a>{'. '}
          Photos from <a href="https://www.pexels.com" target="_blank" rel="noopener noreferrer">Pexels</a>.
        </p>
      </footer>
    </main>
  );
}
