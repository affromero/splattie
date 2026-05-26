import Image from 'next/image';
import Link from 'next/link';
import styles from './page.module.css';

export default function Home() {
  return (
    <main className={styles.hero}>
      <Image src="/logo.svg" alt="Mirada" width={120} height={112} className={styles.logo} />
      <h1 className={styles.title}>Mirada</h1>
      <p className={styles.subtitle}>
        Upload a photo. Get a 3D head whose eyes follow your cursor.
      </p>
      <Link href="/create" className={styles.cta}>
        Try it now
      </Link>
      <Link href="/view/demo" className={styles.demoLink}>
        or see a demo
      </Link>
    </main>
  );
}
