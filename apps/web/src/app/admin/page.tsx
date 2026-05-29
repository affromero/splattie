import Image from 'next/image';
import Link from 'next/link';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { ADMIN_COOKIE, verifySessionToken } from '@/lib/admin-auth';
import { type AdminStats, fetchAdminStats } from '@/lib/admin-stats';
import { LogoutButton } from './LogoutButton';
import styles from './page.module.css';

export const dynamic = 'force-dynamic';

const RANGES = [7, 30, 90];

export default async function AdminPage({
  searchParams,
}: {
  searchParams: Promise<{ days?: string }>;
}) {
  const cookieStore = await cookies();
  if (!verifySessionToken(cookieStore.get(ADMIN_COOKIE)?.value)) {
    redirect('/admin/login');
  }

  const { days: daysParam } = await searchParams;
  const days = RANGES.includes(Number(daysParam)) ? Number(daysParam) : 30;
  const stats = await fetchAdminStats(days);

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>Analytics</h1>
        <div className={styles.headerRight}>
          <nav className={styles.ranges}>
            {RANGES.map((r) => (
              <Link
                key={r}
                href={`/admin?days=${r}`}
                className={r === days ? styles.rangeActive : styles.range}
              >
                {r}d
              </Link>
            ))}
          </nav>
          <LogoutButton />
        </div>
      </header>

      {stats ? <Dashboard stats={stats} days={days} /> : <ErrorState />}
    </main>
  );
}

function Dashboard({ stats, days }: { stats: AdminStats; days: number }) {
  const s = stats.summary;
  return (
    <>
      <section className={styles.cards}>
        <Card label="Visitors" value={s.visitors} hint={`unique · last ${days}d`} />
        <Card label="Pageviews" value={s.pageviews} hint={`last ${days}d`} />
        <Card label="Today" value={s.pageviews_today} hint="pageviews · 24h" />
        <Card label="Last 7 days" value={s.pageviews_7d} hint="pageviews" />
        <Card label="Avatars created" value={s.avatar_creates} hint={`last ${days}d`} />
        <Card label="Avatar views" value={s.avatar_views} hint={`last ${days}d`} />
        <Card label="Editor opens" value={s.editor_opens} hint={`last ${days}d`} />
        <Card label="Bots filtered" value={s.bots} hint={`last ${days}d`} muted />
      </section>

      <section className={styles.panel}>
        <h2 className={styles.panelTitle}>Pageviews per day</h2>
        <BarChart data={stats.timeseries} />
      </section>

      <div className={styles.tables}>
        <Table
          title="Top pages"
          rows={stats.top_paths.map((p) => [p.path, p.views])}
          empty="No pageviews yet."
        />
        <Table
          title="Top referrers"
          rows={stats.top_referrers.map((r) => [r.referrer, r.count])}
          empty="No referrers yet."
        />
        <Table
          title="Devices"
          rows={stats.devices.map((d) => [d.device, d.count])}
          empty="No data yet."
        />
        <Table
          title="Countries"
          rows={stats.top_countries.map((c) => [`${flag(c.country)} ${c.country}`, c.visitors])}
          empty="No country data yet."
        />
      </div>

      <DemoClicks items={stats.demo_clicks} />
    </>
  );
}

function Card({
  label,
  value,
  hint,
  muted,
}: {
  label: string;
  value: number;
  hint: string;
  muted?: boolean;
}) {
  return (
    <div className={muted ? styles.cardMuted : styles.card}>
      <span className={styles.cardValue}>{value.toLocaleString()}</span>
      <span className={styles.cardLabel}>{label}</span>
      <span className={styles.cardHint}>{hint}</span>
    </div>
  );
}

function BarChart({ data }: { data: AdminStats['timeseries'] }) {
  if (data.length === 0) return <p className={styles.empty}>No data yet.</p>;

  const width = 760;
  const height = 160;
  const gap = 3;
  const barWidth = Math.max(2, width / data.length - gap);
  const max = Math.max(1, ...data.map((d) => d.pageviews));

  return (
    <>
      <svg
        className={styles.chart}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Daily pageviews"
      >
        {data.map((d, i) => {
          const barHeight = (d.pageviews / max) * (height - 8);
          return (
            <rect
              key={d.day}
              x={i * (barWidth + gap)}
              y={height - barHeight}
              width={barWidth}
              height={barHeight}
              rx={1}
              fill="var(--color-primary)"
            >
              <title>{`${d.day}: ${d.pageviews} views, ${d.visitors} visitors`}</title>
            </rect>
          );
        })}
      </svg>
      <div className={styles.axis}>
        <span>{data[0].day}</span>
        <span>{data[data.length - 1].day}</span>
      </div>
    </>
  );
}

function Table({
  title,
  rows,
  empty,
}: {
  title: string;
  rows: [string, number][];
  empty: string;
}) {
  return (
    <div className={styles.tableCard}>
      <h2 className={styles.panelTitle}>{title}</h2>
      {rows.length === 0 ? (
        <p className={styles.empty}>{empty}</p>
      ) : (
        <ul className={styles.list}>
          {rows.map(([label, count]) => (
            <li key={label} className={styles.row}>
              <span className={styles.rowLabel}>{label}</span>
              <span className={styles.rowCount}>{count.toLocaleString()}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function flag(code: string): string {
  if (!/^[A-Za-z]{2}$/.test(code)) return '🌐';
  return String.fromCodePoint(
    ...[...code.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)
  );
}

function DemoClicks({ items }: { items: AdminStats['demo_clicks'] }) {
  return (
    <div className={styles.tableCard}>
      <h2 className={styles.panelTitle}>Demo clicks</h2>
      {items.length === 0 ? (
        <p className={styles.empty}>No demo clicks yet.</p>
      ) : (
        <div className={styles.thumbGrid}>
          {items.map((d) => (
            <div key={d.id} className={styles.thumbItem}>
              <Image
                src={`/demos/thumbs/${d.id}.jpg`}
                alt={`Demo ${d.id}`}
                width={56}
                height={70}
                className={styles.thumb}
              />
              <span className={styles.thumbCount}>{d.clicks.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ErrorState() {
  return (
    <div className={styles.errorCard}>
      <p>Could not load analytics from the backend.</p>
      <p className={styles.empty}>
        Check that <code>ADMIN_API_TOKEN</code> matches the backend and that{' '}
        <code>BACKEND_INTERNAL_URL</code> is reachable.
      </p>
    </div>
  );
}
