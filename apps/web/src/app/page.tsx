import Image from 'next/image';
import Link from 'next/link';
import styles from './page.module.css';

export default function Home() {
  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <Image src="/logo.svg" alt="" width={64} height={60} className={styles.logo} />
        <h1 className={styles.title}>Splattie</h1>
        <p className={styles.subtitle}>
          A 3D head from a single photo. Eyes that follow you.
        </p>
        <div className={styles.actions}>
          <Link href="/create" className={styles.cta}>
            Upload a photo
          </Link>
          <Link href="/view/demo" className={styles.secondary}>
            See the demo
          </Link>
        </div>
      </section>
    </main>
  );
}
