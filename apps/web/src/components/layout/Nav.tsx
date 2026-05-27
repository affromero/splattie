'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Nav.module.css';

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className={styles.nav}>
      <Link href="/" className={styles.brand}>
        splattie
      </Link>
      <div className={styles.links}>
        <Link
          href="/view/demo"
          className={pathname === '/view/demo' ? styles.linkActive : styles.link}
        >
          demo
        </Link>
        <Link
          href="/create"
          className={pathname === '/create' ? styles.linkActive : styles.link}
        >
          create
        </Link>
      </div>
    </nav>
  );
}
