import Image from 'next/image';
import Link from 'next/link';
import styles from './page.module.css';

export default function Home() {
  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.left}>
          <div className={styles.logoBlock}>
            <Image src="/logo.svg" alt="" width={48} height={45} className={styles.logo} />
            <span className={styles.label}>SIGGRAPH 2025 / FLAME / 3DGS</span>
          </div>
          <h1 className={styles.title}>
            Single-image head reconstruction with real-time gaze tracking
          </h1>
          <p className={styles.description}>
            Upload a photograph. The system segments the head, reconstructs a 3D Gaussian
            Splatting model, and renders it in your browser. The eyes follow your cursor
            using client-side FLAME animation — no GPU required on the viewer side.
          </p>
          <div className={styles.actions}>
            <Link href="/create" className={styles.cta}>
              Upload a photo
            </Link>
            <Link href="/view/demo" className={styles.secondary}>
              See the demo
            </Link>
          </div>
        </div>
        <div className={styles.right}>
          <div className={styles.demoFrame}>
            <div className={styles.frameLabel}>
              <span>LIVE PREVIEW</span>
              <span className={styles.dot} />
            </div>
            <div className={styles.frameContent}>
              <p className={styles.frameHint}>
                Visit <Link href="/view/demo">/view/demo</Link> for the interactive viewer
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className={styles.pipeline}>
        <span className={styles.sectionLabel}>Pipeline</span>
        <div className={styles.steps}>
          <div className={styles.step}>
            <span className={styles.stepNum}>01</span>
            <span className={styles.stepName}>Segment</span>
            <span className={styles.stepDetail}>SAM 3 in-browser</span>
          </div>
          <div className={styles.stepArrow}>→</div>
          <div className={styles.step}>
            <span className={styles.stepNum}>02</span>
            <span className={styles.stepName}>Reconstruct</span>
            <span className={styles.stepDetail}>LAM on GPU</span>
          </div>
          <div className={styles.stepArrow}>→</div>
          <div className={styles.step}>
            <span className={styles.stepNum}>03</span>
            <span className={styles.stepName}>Compress</span>
            <span className={styles.stepDetail}>SPZ &lt;2MB</span>
          </div>
          <div className={styles.stepArrow}>→</div>
          <div className={styles.step}>
            <span className={styles.stepNum}>04</span>
            <span className={styles.stepName}>Render</span>
            <span className={styles.stepDetail}>FLAME LBS @ 60fps</span>
          </div>
        </div>
      </section>

      <footer className={styles.footer}>
        <span>LAM · SAM 3 · Spark 2.0 · FLAME</span>
        <span className={styles.footerRight}>Andrés Romero · 2026</span>
      </footer>
    </main>
  );
}
