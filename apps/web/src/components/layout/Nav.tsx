'use client';

import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Nav.module.css';

export function Nav() {
  const pathname = usePathname();
  const isHome = pathname === '/';

  return (
    <nav className={styles.nav}>
      <Link href="/" className={styles.brand}>
        <Image src="/logo.svg" alt="Mirada" width={28} height={26} />
        <span className={styles.brandName}>Mirada</span>
      </Link>
      <div className={styles.links}>
        {!isHome && (
          <Link href="/" className={styles.link}>
            Home
          </Link>
        )}
        <Link
          href="/view/demo"
          className={pathname === '/view/demo' ? styles.linkActive : styles.link}
        >
          Demo
        </Link>
        <Link
          href="/create"
          className={pathname === '/create' ? styles.linkActive : styles.link}
        >
          Create
        </Link>
      </div>
    </nav>
  );
}
