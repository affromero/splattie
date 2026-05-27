import styles from './page.module.css';

export default function DemoPage() {
  return (
    <main className={styles.page}>
      <iframe
        src="/demo/viewer.html"
        className={styles.viewer}
        title="3D Head Viewer"
      />
    </main>
  );
}
